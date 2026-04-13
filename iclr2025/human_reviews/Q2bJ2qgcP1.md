## Human Reviewer 1

### Summary
This paper presents a framework to evaluate CATE estimation methods in observational studies. To address challenges in existing approaches that either leverage semi-synthetic data or rely on untestable modeling assumptions, this work proposes to rely on propensity modeling for evaluation. Estimators for the MSE (minus a constant) are proposed and their theoretical properties are discussed under conditions on the propensity models. Then, this paper provides a comprehensive benchmark by deriving observational studies via resampling RCTs, which leads to unbiased evaluation of existing CATE methods. The benchmark reveals surprising findings about the insufficiency of the existing methods for capturing heterogeneity in CATE.

### Strengths
1. This paper studies the important problem of reliable evaluation of CATE methods. 
2. The paper is comprehensive, containing fruitful results. 
3. The paper contributes new tools and benchmarks to the field, which may benefit other researchers and motivate future works. 
4. The paper is clearly structured.

### Weaknesses
1. The writing quality of the paper can still be improved. Sometimes the advantages of the proposed method is a bit overclaimed. It will be helpful to also discuss the limitation of the proposed approach.
2. The discussion on the surprising results can be more in depth. Why are existing methods even worse than constant treatment effect estimation? Is this because they are too variable (so that variance in MSE is large)? Is this because the heterogeneity in the datasets is actually small? 

Please see my questions for more comments.

### Questions
1. Why is Theorem 3.4 better than the double robustness results? It seems to require consistency of the propensity model, but double robustness estimation can be consistent if either the outcome model or the propensity model is correct. 

2. What are the assumptions for Theorem 3.13 to hold? Does the propensity score need to be estimated? If so, why didn't its estimation error enter the result?

3. It's weird to count randomly split data as a "new dataset" so that there are 43200 benchmark datasets. Please consider revise your claim on the number of datasets. 

4. In section 4.3, the performance of $\hat{Q}$ is a bit worse than model-augmented versions of $\hat{Q}$. It would be nice to provide some suggestions on what estimators to use in practice. Does the model-augmented version has additional bias/modeling concerns?

5. What's special about the Horwitz-Thompson estimator for $\eta$? It seems that $\eta$ can be anything whose conditional expectation given $X$ equals $\tau(X)$. Is it the point that it doesn't use any outcome model? Can it be replaced by other reasonable choices? 

6. While addressing the concerns of semi-synthetic evaluation and outcome modeling, here the method requires the propensity model to be well estimated (though I understand that the evaluation part doesn't have this concern due to controlled sampling process). Is this over-simplifying real-world evaluation scenarios where the treatment assignment in observational studies can be complicated as well? What is the method evaluating if an inconsistent propensity model is used? Is it estimating some other quantity that is related to MSE of CATE estimator? 

7. I would suggest separating the theoretical results for the most general evaluation case (eg in a real observational study) and the evaluation part in this paper. In real observational studies, consistent evaluation inevitably rely on some good modeling. However, it is important to be clearly stated that the evaluation in this paper doesn't need this because of the controlled sampling process.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper conducts an empirical study of the accuracy of CATE estimation methods, asking whether existing methods in the literature reliably outperform trivial benchmarks. The main idea is to use RCT datasets, where the propensity is known, to construct benchmarks where a simulated propensity score is used to construct an "observational" sample. Since the propensity score is known for the original data, estimators on the observational sample can be benchmarked without having to rely on accurate estimates of propensity or outcome models (effectively using a HT estimator, potentially augmented with other regression models just for variance reduction purposes).

### Strengths
This work is definitely important for the field. Evaluating CATE models is very difficult due to the risk of self-serving bias that the paper discusses, and it is unclear when in practice modern techniques are helpful. Constructing benchmarks and evaluation methods which use only real data, via RCT datasets, is important. Using semi-synthetic datasets with simulated outcomes, as is common in the field, is much less convincing.

### Weaknesses
There are some respects in which the execution of the paper could be improved:

(1) Based on plots in the appendix, there appears to be significant variation in the performance of different methods across datasets (as would be expected). This deserves more discussion, and investigation. E.g. why do double ML methods perform well in some cases and not in many others? One hypothesis, based on benchmarks for ATE estimation, would be that outcome regression methods work best as long as the outcome models are reasonably well estimated, with double-ML style methods providing benefits when the outcomes are not as well captured. 

As is, there is little in the way of takeaways about the conditions under which existing methods do or don't perform well. This sort of diagnosis would be at least as useful as the headline results about % of cases where methods work well or don't, since the headline numbers are very specific to the composition of this specific benchmark. 

(2) Relatedly, the benchmark appears to be weighted quite heavily towards a set of multiple tasks all drawn from the same dataset: out of 12 tasks, 8 are different outcomes taken from the general social survey. We might reasonably expect different tasks from the same dataset to share some similar characteristics so the amount of actual diversity in the benchmark is less than what the number of tasks would suggest. 

(3) Regarding the degeneracy rate, are the results different if we test for whether Q is significantly larger than 0 (i.e., a CI excludes 0) rather than if the point estimate is > 0? I would guess that most of the datasets are large enough for the estimates of Q to be fairly precise, but it would be helpful to verify this. 

While I think the paper makes a worthwhile contribution even with these weaknesses, the contribution would be significantly strengthened with a more diverse set of tasks and more analysis of what drives performance variation across those tasks.

### Questions
Any comments/clarifications, particularly related to points 1 and 3 above, would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents a large-scale benchmark study evaluating the performance of contemporary Conditional Average Treatment Effect (CATE) estimation models. The authors use a novel application of observational sampling to evaluate 16 modern CATE models across 43,200 datasets.  Their key findings challenge the effectiveness of current CATE models in capturing real-world heterogeneity.

### Strengths
Large-scale and comprehensive benchmark study.

Novel approach to CATE evaluation using observational sampling and the Q statistic.

Rigorous theoretical analysis and proofs supporting the proposed methodology.

Use of real-world datasets provides valuable insights into the limitations of current CATE models.

### Weaknesses
While aiming for diversity, the selection of datasets may still not fully represent the breadth of real-world applications, potentially limiting the generalizability of the findings. More explanation of dataset selection criteria would strengthen the paper.

The paper focuses on MSE as the primary evaluation metric. While justified by its practical relevance, exploring alternative evaluation metrics could provide additional insights.

While innovative, the reliance on the Q statistic is new and requires further validation and adoption by the wider research community.

### Questions
The paper discusses the generalization of the Q statistic to new distributions. Could you provide more details about the assumptions underlying these generalization results? How sensitive are these results to violations of these assumptions?

What are the main limitations of the current study, and what directions for future research do you suggest based on these findings? What kinds of new CATE models or evaluation methods should be explored? What types of additional data would be useful for further validation?

How can the findings of this study be used to guide the practical application of CATE models in various domains (medicine, economics, etc.)? What advice would you give to practitioners regarding the selection and interpretation of CATE models?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
In this work, the authors propose a new way to estimate treatment effect and perform model selection when the counterfactual is not available. To do this, the authors propose a new metric called Q, which is derived from the Mean Squared Error (MSE) but adjusted to exclude terms that do not depend on the estimator.

### Strengths
Lacking good metrics for evaluation is a big problem in casual inference due to no access to counterfactuals. This paper proposes a new metric, which could be helpful for practitioners when comparing different models.

### Weaknesses
1. For the experiments given in this work, the estimation (training) size is much smaller than the evaluation size. This is not intuitive to me as in typical ML settings, we have training-test ratio to be 4:1. Using a much lower ratio of training vs test samples could potentially lead to the problem of unfitting, which could be a reason why many baselines have too many degenerate cases. I think it is necessary to use a large number to training samples to rule out this hypothetical scenario.

### Questions
1. Can you explain why your estimation size is much smaller than the evaluation size?

2. Can you re-conduct the experiments when the estimation size is much larger than the evaluation size? Let's use the more standard ML experimental setup, where the ration between training and test samples is 4:1. You should apply this to all datasets in your Table 4. Then rerun the whole experimental pipeline and report results similar to those shown in Table 1.

3. Also, there is no standard deviation for Table 1. It might be a little bit misleading in using the description "43,200 datasets" in Table 1. What you are essentially doing is repeated resampling, similar to bootstrap. In this case, in addition to what I ask for #2 above, could you also report results on the same datasets, with same percentage of treatment and other parameters fixed, with both mean and standard deviation, where the standard deviation solely comes from bootstrap?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 5

### Summary
The paper proposes a new statistical parameter to evaluate the performance of a method to estimate CATE. Then the paper shows that this parameter can be consistently estimated. Finally the paper demonstrates that this parameter can be used to select estimation methods for CATE.

### Strengths
- The statistical parameter proposed in the paper is simple to implement and makes intuitive sense. 
- Theoretical guarantees are provided. 
- The paper compares many estimation methods for CATE in the empirical application.

### Weaknesses
- The paper largely overclaims the number of datasets. Essentially, there are only 12 unique datasets and the paper uses a bunch of sampling methods to create many more variants of the original 12 datasets.
- The findings in the first paragraph of the abstract have been known for a long time, and they are not unexpected, as claimed in the paper. The estimation of CATE is known to be very noisy, and that's why the causal inference literature has primarily focused on the estimation of ATE (but not CATE) for decades. 
- The clarity and rigor of the paper need to be improved. For example, the term "CATE model" is a bit awkward and is rarely used in the literature. CATE is an estimand commonly defined based on potential outcomes. When "CATE model" is used, it is unclear whether this term refers to an outcome model to define CATE or an estimation method for CATE.

### Questions
Please address the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3