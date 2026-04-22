# More Bang for the Buck: Process Reward Modeling with Entropy-Driven Uncertainty

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
We introduce the Entropy-Driven Uncertainty Process Reward Model (EDU-PRM), a novel entropy-driven training framework for process reward modeling that enables dynamic, unce
rtainty-aligned segmentation of complex reasoning steps, eliminating the need for costly manual step annotations.
Unlike previous Process Reward Models (PRMs) that rely on static partitioning and human labeling, EDU‑PRM automatically anchors step boundaries at tokens with high predictive entropy, effectively capturing intrinsic logical transitions and facilitating efficient exploration of diverse reasoning paths.
On the ProcessBench benchmark, EDU-PRM outperforms strong public PRM baselines, such as Math-Shepherd PRM and Omega PRM, and EDU-PRM achieves comparable results with SOTA models while only using 1.5\% training data.
Furthermore, by leveraging our proposed EDU sampling strategy, we observe accuracy boosts from 64.7\% to 67.3\% for generative reasoning tasks, accompanied by a reduction of 32\% in token usage.
These findings underscore the potential of EDU-PRM as a scalable and annotation-efficient paradigm for process supervision in mathematical reasoning, paving the way for more efficient and robust approaches to complex mathematical problem solving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the Entropy-Driven Uncertainty Process Reward Model (EDU-PRM). The core idea is to use token-level predictive entropy to dynamically and automatically segment reasoning steps. The method identifies high-entropy tokens as "uncertainty anchors," which are assumed to mark natural logical transitions. 

The PRM trained on this automatically generated data outperforms strong baselines on the ProcessBench benchmark. They also show their inference sampling strategies provide higher accuracy for fewer tokens compared to standard high-temperature sampling.

### Strengths
1. The core contribution—using predictive entropy to find "uncertainty anchors" for segmenting reasoning is a well-motivated alternative to arbitrary, rule-based partitioning. 

2. A very comprehensive experiments shows the performance of the new algorithm.

### Weaknesses
1. The novelty of the core idea in this paper is not a fundamental breakthrough, as it is built based on a few previous work.

### Questions
1. ' the training dataset comprises approximately 1.42M instances, with a label distribution of 52% hard and 48% soft labels.' How the hard and soft labels are generated? I assume they should be the same?

2. When you run the comparison with other PRM models, do you use the data provided or you implement the algorithm by yourselves?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EDU-PRM, a process reward model that uses token-level entropy to identify key decision points in reasoning. It segments reasoning chains at high-entropy tokens and uses MCE to assign reward scores to each segment based on outcome correctness. The experimental results show that EDU-PRM achieves significant performance improvements on ProcessBench and multiple math reasoning benchmarks while reducing token consumption.

### Strengths
1. The paper is well-motivated, clearly written, and easy to follow.
2. The idea of using high-entropy tokens to segment reasoning steps is simple yet intuitive, effectively reducing the reliance on manual annotations or LLM-based heuristics.
3. Extensive experiments across multiple benchmarks demonstrate the effectiveness of the proposed method.

### Weaknesses
1. While the paper claims novelty in introducing entropy into process reward modeling, entropy-based approaches have been explored previously. The authors do not sufficiently discuss their method with closely related work, such as:

    ○ Entropy-Regularized Process Reward Model

    ○ Uncertainty-Aware Step-wise Verification with Generative Reward Models

    ○ Uncertainty-Based Methods for Automated Process Reward Data Construction and Output Aggregation in Mathematical Reasoning

2. The definition and application of the entropy threshold $\tau(H)$ are confusing and under-specified. In Section 3.2, the authors claim $\tau(H)$ is dynamically adjusted based on the maximum number of sampled branches, suggesting an adaptive design. Yet, no formula, algorithm, or implementation detail is provided. More confusingly, in Section 4.1, the authors mention using a fixed entropy threshold (entropy threshold = 1.0), and in Section 5 (Tables 4 and 5), their analysis is also conducted around different fixed threshold values. This inconsistency makes the thresholding mechanism conceptually vague, weakening the clarity and reproducibility of the method.
3. The paper claims that EDU-PRM alleviates the issue of “cheating”, where high intermediate rewards do not necessarily correlate with correct final answers. However, its Monte Carlo Estimation Scoring (MCE) still relies on the correctness of the final answer to assign credit to intermediate segments. Consequently, if an incorrect reasoning step accidentally leads to a correct final answer, MCE may still assign it a high reward. Although entropy-based segmentation improves over heuristic splitting, the MCE mechanism itself does not fundamentally resolve this problem, and the paper should discuss this limitation more explicitly and cautiously.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new sampling method for (i) generating a new dataset for PRM training and (2) test-time PRM inference. This method works by spanning parallel generations for tokens with high predictive entropy only (the “anchors”) and estimating the score for sub-trajectories between two anchor points using Monte Carlo estimation. The paper provides empirical results for the accuracy of PRMs on ProcessBench, as well as BoN results following the resultant PRMs and further analysis on scaling trends, branching, lexical analysis, etc.

### Strengths
- The method seems to have promising gains on Best-of-N evaluation of the proposed PRM-72B model.

- The paper brings a rich experimental analysis, both quantitatively and qualitatively.

### Weaknesses
- The major concern is that the quantitative experiments do not bring any evidence of statistical significance. There are no error bars in any of the experiments. The paper states running 8 experimental seeds, but it only reports the average. From only the average it is unclear if the reported gains are meaningful or just observation noise.

- It is also unclear if training all the baselines on the dataset generated by the EDU sampling (as stated in L258-259) is indeed a fair choice. The dataset is inherently biased towards the proposed heuristic and may benefit the proposed method

- It is also not clear what is the methodological contribution in the paper. The Related Work section is superficial and does not contrast with the literature in the area, being limited to the two baselines adopted. My understanding is that the work has limited novelty, as using uncertainty (or confidence) heuristics to guide inference is an explored direction  [1, 2, 3], including in the specific context of PRMs [4].

- The results related to high temperature sampling in the paper are limited for T = 0.7. It is important to analyze the impact of different temperatures in the method, as this is a quite sensitive hyperparameter. Furthermore, there is no mention on how the hypers of the proposed and comparison methods were selected. It is unclear if the tuning procedure was fair among the methods.

More specific weaknesses:
- In Table 1, the 7B EDU-based models present a very low recall / F1 score. From my understanding of the problem setup, it means the PRMs are failed to identify several wrong steps. In any case, the paper should discuss the reason behind this, as the current takeaway is that the proposed method seems to fail for the 7B scale.

- Nit: The paper describes that there is a Symbol set to avoid mathematical symbols (e.g., sum, integral) in the entropy calculation. But the list in Appendix A.4 does not bring the aforementioned math symbols.

- During Introduction, the paper motivates the proposed method as a way to prevent “cheating” vulnerabilities in PRMs. The paper does not make clear how the method prevent such vulnerabilities, nor any of the experiments linked back to this claim. It is unclear if the proposed PRMs really addressed the defined issue, as the experiments only concentrate on final accuracy.

- It is also hard to take any conclusion from Figure 5 when comparing EDU and P-EDU. Besides the lack of error bars, the curves look nearly identical, and the discussion is limited to analyzing specific points in the curves to make more general claims. My takeaway from the Figure is that P-EDU does not make a meaningful change in the final result, which seems to be the opposite to what is claimed.

A final note: while I appreciate the efforts on providing a rich set of experiments, the paper discussion needs more polishing to condense the information in clear claims/takeaways. There are many different setups/metrics/analyses and it is crucial to better map experiments to specific claims and highlight them. Currently it is hard to filter this information, and the claims are in general vague (e.g., “This highlights EDU sampling’s superior capability to leverage additional tokens for sustained accuracy gains.” in L352).

### Questions
- What are the Qwen2.5-Math-PRM results for the BoN setting?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The paper proposes EDU-PRM, a new way to train process reward models using token-level entropy to detect where a model is uncertain during reasoning.
- This entropy-driven uncertainty automatically marks step boundaries and creates diverse reasoning paths without human or LLM annotations.
- Each reasoning fragment is labeled automatically through Monte-Carlo estimation based on whether the final answer is correct.
- The resulting EDU-PRM matches or nearly matches the performance of the large, fully supervised Qwen2.5-Math-PRM.
- An enhanced version called Pruning-EDU further improves efficiency by cutting off low-confidence reasoning paths early, reducing token use with minimal accuracy loss.

### Strengths
- It eliminates the need for human or LLM annotations by automatically labeling reasoning steps using entropy and Monte Carlo estimation.
- The entropy-based segmentation captures natural reasoning boundaries, improving how step-level rewards relate to final correctness.
- The pruning and entropy-guided sampling reduce token usage and search complexity compared to exhaustive sampling or MCTS.

### Weaknesses
- All experiments are restricted to math reasoning, so generalization to other domains remains unproven.
- The Monte Carlo estimation can produce imperfect or misleading correctness scores for intermediate steps.
- Performance depends on carefully tuning the entropy threshold that defines where to branch or segment reasoning.

### Questions
- How do you choose the entropy threshold for step segmentation?
- Since Monte Carlo estimation relies on final-answer correctness, how do you mitigate cases where an incorrect final answer arises from an otherwise correct partial reasoning step?
- How do you decide the pruning threshold (e.g., PRM score < 0.2)?
- Since your evaluation focuses on math, how could EDU-PRM be adapted to domains where final correctness cannot be automatically verified (e.g., commonsense or scientific reasoning)?
- Can you share more qualitative examples where high-entropy points clearly align with human-intuitive reasoning transitions?
- The authors show Gaussian-smoothed trends. Can you share the raw results before smoothing to see the original values?
- In Line 13, unce rtainty-aligned -> uncertainty-aligned

### Soundness
3

### Presentation
2

### Contribution
3
