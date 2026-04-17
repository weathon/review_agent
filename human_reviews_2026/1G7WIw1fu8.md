# Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has become a key technique for enhancing LLMs' reasoning abilities, yet its data inefficiency remains a major bottleneck. To address this critical yet challenging issue, we present a novel gradient-alignment-based method, named \textit{LearnAlign}, which intelligently selects the learnable and representative training reasoning data for RLVR post-training. To overcome the issue of response-length bias in gradient norms, we introduce the data learnability based on the success rate, which can indicate the learning potential of each data point. Experiments across five reasoning benchmarks show that our method significantly reduces training data requirements while achieving minor performance degradation or even improving performance compared to full-data training. Specifically, it reduces data requirements by up to 1,000 data points with better performance (77.5$\%$) than that on the full dataset on the GSM8K benchmark (77.0$\%$). Furthermore, its efficiency is demonstrated on both mathematical and code benchmarks by using much less data from the DAPO-MATH-17K dataset. We believe this work provides some insights for data-efficient RL post-training and could help future research on reasoning data selection. To facilitate future work, we will release code.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LearnAlign, a data selection method that combines gradient information and successful rate. The experiments show that using only 1,000 samples selected by this method can achieve comparable performance than training on the full dataset, and can outperform other data selection baselines on multiple test sets.

### Strengths
1. The idea of using gradient information to select representative data has certain theoretical significance.
2. The dataset obtained through this method is greatly reduced in size compared to the original dataset while achieving comparable performance.
3. The paper provides insights of how traditional SFT data selection methods perform in RLVR.

### Weaknesses
1. Lack of a clear comparison of convergence speed. Simply comparing results at the level of epochs may not be sufficient. Table 2 shows that the full dataset achieves a better final result. A natural question is: Under the same number of training steps, how does the performance of the full dataset compare to LearnAlign?  A curve of val_score versus training step would help visualize and compare the convergence speed on the full dataset and on the selected dataset.

2. Lack of discussion of training convergence behavior and the final performance ceiling. In general, a larger dataset can support longer training and provide a better final result. Can the selected data, by training for more epochs, eventually reach the same performance level as the full dataset?

3. The method relies on a warm-up training phase to obtain gradient information. However, in RLVR the model’s output distribution typically shifts substantially during training, so the initial gradient information may not reliably indicate which samples are useful. Moreover, the warm-up phase requires fully processing the entire dataset and computing the LearnAligner score, which has O(n²) complexity, leading to significant computational overhead.

4. Missing key baseline. A very common practice in RLVR is to use the pass@N score to filter data. For example, one can drop questions whose pass@8 score is 0/1/7/8, because such questions are either too easy or too hard. Since this approach is extremely easy to implement and is already used as a filtering step before many RLVR training pipelines, the authors should consider comparing against this basic method.

### Questions
Please consider addressing the issues raised under Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the significant data inefficiency of Reinforcement Learning with Verifiable Rewards (RLVR) used for enhancing LLM reasoning. The authors propose a novel data selection method called LearnAlign, which selects a small, high-quality subset of learnable and representative data for training. Experiments across reasoning benchmarks show that LearnAlign significantly outperforms existing data selection methods.

### Strengths
* The paper tackles a practical and important bottleneck in post-training LLMs: the high cost and data inefficiency of RLVR.
* The paper is well-written and easy to follow.
* The data selection process defined in the paper is quite reasonable and logically structured.

### Weaknesses
* The proposed data learnability takes the form of a simple quadratic function. While using the success rate (p) as a basis for the metric is logical, it is questionable whether the metric had to be in this specific multiplicative form. The method would be more persuasive if sufficient justification or additional experiments (e.g., exploring alternative functions) were provided for this design choice.
* Since the entire data selection process is determined by the warm-up model, there is a concern that it might be overly dependent on the specific random dataset chosen for this warm-up. Although the authors showed in Table 3 that warm-up training is a necessary component, it is unclear how sensitive the selection process is to the randomness of this initial dataset.

### Questions
* This paper seems to propose a method for learning well by selecting representative data from the dataset. How does this focus on representativeness affect the learning process in terms of diversity? What are the authors' opinions on the trade-off and balance between selecting representative data versus diverse data?
* Regarding the results in Table 4, what are the computational costs for the other baselines, such as the PPL-based methods or SelectIT?

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
3

### Summary
This work investigates data selection strategy for RLVR training based on gradient estimation for each training datapoint. To estimate the contribution of each training sample, a score for each pair of datapoints are computed that takes into account the pass rate of and the low rank estimation of the gradients on the datapoints. Finally, the top-N datapoints with the highest scores are selected as the training dataset. This approach successfully identifies valuable subsets that could achieve competitive performance as the full training set. Comparison with different data selection approach also shows that the proposed method outperforms existing approaches.

### Strengths
1. The paper is well written and description of the method is easy to follow.
2. The experiment section is comprehensive, covering performance comparison with baseline methods, ablation study, and time cost analysis.
3. Besides the time cost experiment, the paper also provides a theoretical analysis on the data collection cost.

### Weaknesses
See questions.

### Questions
1. While the proposed approach has shown to be effective according to the experiment results, is there any theoretical analysis on why the proposed approach work?
2. How could the gradient information estimation step be implemente efficiently?
3. In Step 4,the LearnAlign score matrix is extremely large since n is the size of training dataset. How is this step implemented efficiently?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces LearnAlign, an efficient data selection method for RL post-training, particularly in RLVR (Reinforcement Learning from Verifiable Rewards) settings. LearnAlign aims to improve data selection efficiency by incorporating a metric that is scaled by the success rate of each data point, in addition to considering the cosine similarity between the gradients of data.

### Strengths
- The authors propose a novel "Data Learnability" metric, which utilizes success rates to identify which data points possess more significant gradients. The experiments provided demonstrate that this metric can lead to more effective data selection.

### Weaknesses
- The novelty of LearnAlign appears limited when compared to LESS [1]. The core architecture seems largely similar, with the main difference being the use of verifiable rewards to compute a success rate, which in turn is used to scale the data importance metric. 
- The empirical results in Tables 1-3 are presented solely as average performance. To properly evaluate the statistical significance of the claimed improvements, it is recommended that the authors also report standard deviations or confidence intervals across multiple runs.
- Based on the results in Table 3, the contributions of key components like "warmup" and "data learnability" do not appear to be significant. The "data learnability" metric is presented as a central component of the proposed method, yet its inclusion seems to yield minimal performance benefits, which raises questions about its practical impact.

[1] M. Xia, et al., “LESS: Selecting Influential Data for Targeted Instruction Tuning,” ICML 2024.

### Questions
Please refer to the points raised in the Weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
1
