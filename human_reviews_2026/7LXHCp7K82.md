# EWMV: An algorithm to improve the efficiency of conformal methods

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Conformal prediction is a framework that augments machine learning models to return a prediction set in lieu of a single prediction. Although, these sets commonly contain the correct answer with guaranteed probability, their size can be ineffectively large and thus lead to costly erroneous decisions. To mitigate this, we propose EWMV, an algorithm that leverages the available calibration data to aggregate multiple accessible predictors into a single, smaller conformal predictor. Empirical evidence across a variety of tasks and conformal methods suggests EWMV often produces smaller and more efficient prediction sets than any of the individual predictors being aggregated. Accordingly, these findings encourage a new paradigm to improve the efficiency of conformal methods with two readily available resources: calibration data and a plethora of pre-trained predictors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This submission proposes to minimize the average size of a CP-based prediction set over the weights used to combine different CP prediction sets. The approach is based on minimizing an empirical estimate of the average set size that is based in held-out data. Computationally, optimization is done either addressing a brute-force mixed integer linear program (MILP) formulation or using a standard linear program (LP) approximation obtained by replacing the indicator function with a hinge function. 

A theoretical result is provided that demonstrates a sufficient condition under which the optimal weights do not increase the set size.

Experimental results are given for various tasks.

### Strengths
While the contribution is technically minor (see below), the submission presents some potentially interesting results on the benefits of CP set combination. 

The paper is relatively clear, although the quality of the text could be significantly improved.

### Weaknesses
The proposed approach appears conceptually straightforward. The authors reserve a portion of the data and then optimize the prediction set size using this held-out subset. While this specific strategy may not have been explored in prior work, it does not seem to offer any formal guarantees. In particular, the main result concerning the expected set size (Proposition 5.2) states that if the optimal predictor’s prediction set size remains unchanged when increasing the coverage requirement, then optimizing the weight selection cannot reduce the average size. This conclusion is rather self-evident and does not directly address the challenges inherent in finite-sample, held-out data settings. Moreover, the paper does not clarify under what practical conditions the assumed invariance property of the optimal predictor would hold.

A further concern relates to data efficiency. Would it not be preferable to use the held-out data directly for calibration instead? In this context, one might have expected the authors to explore a meta-learning approach for weight optimization, allowing the procedure to rely solely on data from other tasks and thus avoid the need for task-specific held-out data.

### Questions
1. Can performance guarantees be provided that account for the use of finite held-out data and that do not require the strong assumption in Proposition 5.2?

2. Can the method be generalized to apply meta-learning across multiple tasks to avoid the need for held-out data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
**Summary**

Conformal prediction provides a prediction sets that contain the true label with a guaranteed probability, but these sets are often inefficiently large, limiting their practical use. 
This paper introduces EWMV (Estimating Weighted Majority Vote), an algorithm designed to address this inefficiency. 
EWMV aggregates multiple conformal predictors into a single, more compact one. 
It leverages the Weighted Majority Vote (WMV) framework, where a label is included in the final set if the sum of weights from predictors containing that label surpasses a threshold. 
The core innovation of EWMV is its data-driven method for finding optimal weights. It uses a portion of the calibration data (an "estimation set") to learn weights that explicitly minimize the average prediction set size. 
This learning problem is formulated as an optimization task, solvable via Mixed-Integer Linear Programming (MILP) for exactness or a faster 
By constructing the base predictors at a more conservative error level (e.g., $\alpha/2$), the final aggregated predictor is guaranteed to maintain the desired coverage of at least $1-\alpha$.

**Contribution**

The paper's main contribution is an algorithm that improves the efficiency of conformal methods without sacrificing their core validity guarantees. 
It provides a practical solution to the critical problem of selecting weights in the WMV framework, which was previously a major obstacle. 
This work establishes a new paradigm for enhancing conformal prediction: instead of just selecting the best single model, it demonstrates that aggregating multiple readily available pre-trained models can yield superior results. 
The authors provide theoretical justification for the algorithm's validity and present a sufficient condition for its efficiency gain over the best individual predictor.

### Strengths
**Originality**
The paper introduces EWMV, an algorithm for aggregating conformal predictors. Its primary originality lies in proposing a data-driven approach to learn the aggregation weights for the Weighted Majority Vote (WMV) framework. This addresses a key limitation of prior work where weight selection was often heuristic or suboptimal. By formulating the weight estimation as an empirical risk minimization problem solvable with standard optimizers (MILP or LP), the work provides a practical and principled method for improving the efficiency of prediction sets.

**Quality**
The work is well-grounded in the existing literature on conformal model aggregation. It clearly contextualizes its contribution by discussing and differentiating from prior p-value combination methods and set-based approaches. The paper builds upon the theoretical foundations of WMV to ensure the validity of the proposed aggregation scheme. This thorough background demonstrates a clear understanding of the field and helps to precisely situate the paper's novel contribution, strengthening the overall quality of the research.

**Significance**
The empirical evaluation is a significant strength of the paper. The authors validate their algorithm across a diverse set of four tasks, including image classification, question answering, and risk stratification, using multiple underlying conformal methods. This extensive testing demonstrates the general applicability and robustness of EWMV. The results consistently show that the proposed method improves efficiency over individual predictors and simpler aggregation baselines, providing strong evidence for the algorithm's practical utility.

### Weaknesses
The paper's experimental comparison could be strengthened by including other recent and concurrent work on the aggregation and selection of conformal prediction sets. Methods from Ge et al. (2024), Liang et al. (2025), Luo and Zhou (2024), Qin et al. (2024), and Yang and Kuchibhotla (2024) all address related problems. Comparing EWMV against these approaches would provide a more complete picture of its performance and position within the current literature, clarifying its relative advantages in efficiency and applicability.

A more detailed discussion of the methodological gap with score-level aggregation methods would be beneficial. For instance, Luo and Zhou (2024) also propose learning weights, but they aggregate the non-conformity *scores* before computing the quantile. In contrast, EWMV aggregates the final prediction *sets* after the quantile step for each predictor. Clarifying this distinction—aggregating pre- vs. post-conformalization—would help readers better understand the unique contribution and specific use case of the EWMV approach.

The weight estimation procedure on the estimation set $D_{nest}$ can be viewed as a form of tuning. As explored by Zeng et al. (2025) in other contexts, such data-dependent tuning can introduce a "tuning bias." The weights learned on a finite $D_{nest}$ may be slightly overfit, potentially leading to sub-optimal efficiency on unseen test data. The paper would be more complete if it acknowledged and briefly discussed this potential source of bias and how it might affect the generalization performance of the aggregated predictor.

#### References
Ge, Jiawei, Debarghya Mukherjee, and Jianqing Fan. 2024. “Optimal Aggregation of Prediction Intervals under Unsupervised Domain Shift.” 37 (November). https://openreview.net/forum?id=ldXyNSvXEr.

Liang, Ruiting, Wanrong Zhu, and Rina Foygel Barber. 2025. “Conformal Prediction after Data-Dependent Model Selection.” arXiv:2408.07066. Preprint, arXiv, July 3. https://doi.org/10.48550/arXiv.2408.07066.

Luo, Rui, and Zhixin Zhou. 2024. “Weighted Aggregation of Conformity Scores for Classification.” arXiv:2407.10230. Preprint, arXiv, July 14. https://doi.org/10.48550/arXiv.2407.10230.

Qin, Shenghao, Jianliang He, Bowen Gang, and Yin Xia. 2024. “SAT: Data-Light Uncertainty Set Merging via Synthetics, Aggregation, and Test Inversion.” arXiv:2410.12201. Preprint, arXiv, October 16. https://doi.org/10.48550/arXiv.2410.12201.

Zeng, Hao, Kangdao Liu, Bingyi Jing, and Hongxin Wei. 2025. “Parametric Scaling Law of Tuning Bias in Conformal Prediction.” Paper presented at International Conference on Machine Learning. Forty-Second International Conference on Machine Learning, June 18. https://openreview.net/forum?id=jnJLZXSOin.

### Questions
1. The experimental evaluation could be strengthened by including more recent baselines. Could you comment on how EWMV compares to other conformal aggregation and selection methods, such as those proposed by Yang & Kuchibhotla (2024) or Luo & Zhou (2024), and consider adding them to the empirical study?

2. The weight estimation on the set $D_{nest}$ is a form of data-dependent tuning. Could you discuss the potential for "tuning bias" in this step if $D_{nest}$ = $D_{cal}$? Specifically, how might the finite size of the estimation set affect the generalization of the learned weights and the efficiency of the final aggregated predictor on unseen data?

3. Could you make difference between EWMV and score-level aggregation approaches like that of Luo & Zhou (2024)? What are the key trade-offs between aggregating the final prediction sets (post-conformalization) versus aggregating the non-conformity scores (pre-conformalization)?

4. The paper focuses on classification. Have you considered extending the EWMV framework to regression problems for aggregating prediction intervals (such as Ge et al. (2024))? What would be the primary challenges in adapting the weighted majority vote mechanism and the optimization objective to handle continuous outputs? 

I believe that responses to some or all of these points would significantly strengthen the paper, and I would be willing to increase my rating accordingly.

### Soundness
3

### Presentation
4

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
This paper introduces EWMV (Efficient Weighted Majority Vote), an algorithm designed to aggregate multiple conformal predictors into a single, more efficient predictor that produces smaller prediction sets while preserving validity. Conformal prediction provides uncertainty quantification by outputting prediction sets that contain the true label with a guaranteed probability ($1-\alpha$). The authors leverage calibration data to estimate optimal weights for a weighted majority vote aggregation via linear programming (LP) or mixed-integer linear programming (MILP) formulations. Theoretical guarantees include validity preservation and a sufficient condition for efficiency improvement. Empirical evaluations across synthetic multiclass classification, image classification, natural question answering, and risk stratification demonstrate that EWMV often yields smaller sets than individual predictors or baselines like majority vote.

### Strengths
The paper presents new algorithms for weights computation for CP based on majority voting. Experimental results are extensive, demonstrating the performance on diverse tasks. Computational complexity seems reasonable.

### Weaknesses
1. The novelty of the proposed method is somewhat limited as it strongly relies on previous works on weighted majority voting, with the main difference of proposing practical algorithms for weight optimization. 

2. Theoretical statements are either grounded in prior work (as in Proposition 5.1) or depend on assumptions that may be excessively strong (as in Proposition 5.2).

3. The paper does not refer to another approach for conformal ensembling at the predictor level, via score weighting or multi-dimensional CP. See for example:
[1] Bai, Yu, et al. "Efficient and Differentiable Conformal Prediction with General Function Classes." International Conference on Learning Representations (2022). - see Appendix F
[2] Luo, Rui, and Zhixin Zhou. "Weighted aggregation of conformity scores for classification." arXiv preprint arXiv:2407.10230 (2024).‏
[3] Tawachi, Yam, and Bracha Laufer-Goldshtein. "Multi-dimensional conformal prediction." The Thirteenth International Conference on Learning Representations. 2025.‏

### Questions
Please address the issues outlined in the Weaknesses section, as well as the additional questions listed below:
1. Proof of proposition 5.2 - why is the first inequality valid? Eq. (5) is an optimization on an empirical estimate, while the proposition deals with the actual expectation. In addition, the assumption "if the average size of the most efficient predictor does not change when we re-estimate it at a more conservative level” does not seem to hold in practice as it almost always the case that the set size increases for more conservative levels, and especially when decreasing $\alpha$ by a factor of $2$.
2. How does the method work with other nonconformity scores, such as $1-p(y|x)$?
3. Comparisons to other baselines is only shown for CIFAR-100 with $\alpha=0.05$ (which is different from $\alpha=0.1$ used for the same experiment in Table 2. Could you provide additional comparisons  over other datasets/tasks and other coverage levels?
4. line 419: “For reference, we also plot the runtime of MD (From section 6.1)” - what is MD?
5. Figure 7 is unclear - why is MILP performance fixed across the number of models - can you show the performance as a function of the number of heads for both proposed methods (MILP and LP) as well as other baselines?
6. Minor - “We use RAPS from Angelopoulos et al. (2022) as the conformal method and obtain all the fine-tuned models along with the dataset are available from HuggingFace and Torchvision” - “are available” seems redundant

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
2

### Summary
The paper proposes an approach, EWMV, that aggregates the predictions of multiple conformal predictors into a single prediction set. The authors show through empirical investigation that EWMV usually generates more efficient prediction sets than the aggregated predictors. The evaluations also included a range of tasks and multiple conformal prediction methods.

### Strengths
- The paper is proposing a novel approach that improves the informational efficiency of predictions.
- The evaluation is thorough and has been conducted across multiple tasks.
- The results show efficiency enhancement with EWMV compared to the competing approaches.
- Overall, the paper can be followed and understood with relative ease.

### Weaknesses
- The paper provides an incremental improvement over the weighted majority voting algorithms. In other words, the novelty is limited.

- The computational complexity is a drawback of the proposed approach (exponential for MILP).

- Some acronyms were used in the text before being introduced.

- In section 6.7, coverage needs to be clearly defined.

- There is also a typo in the first sentence of the abstract.

### Questions
1- In what sense sections 6.6 and 6.7 provide ablation studies? i.e., what part of EWMV has been removed to study its effect on the performance? Section 6.6, for instance, looks like a classic comparison between different methods.

2- Line 426 starts with "In the table", which table?

### Soundness
3

### Presentation
2

### Contribution
2
