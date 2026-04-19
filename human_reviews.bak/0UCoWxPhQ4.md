# SAVA: Scalable Learning-Agnostic Data Valuation

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Selecting data for training machine learning models is crucial since large, web-scraped, real datasets contain noisy artifacts that affect the quality and relevance of individual data points. These noisy artifacts will impact model performance. We formulate this problem as a data valuation task, assigning a value to data points in the training set according to how similar or dissimilar they are to a clean and curated validation set. Recently, *LAVA* (Just et al., 2023) demonstrated the use of optimal transport (OT) between a large noisy training dataset and a clean validation set, to value training data efficiently, without the dependency on model performance. However, the *LAVA* algorithm requires the entire dataset as an input, this limits its application to larger datasets. Inspired by the scalability of stochastic (gradient) approaches which carry out computations on *batches* of data points instead of the entire dataset, we analogously propose *SAVA*, a scalable variant of *LAVA* with its computation on batches of data points. Intuitively, *SAVA* follows the same scheme as *LAVA* which leverages the hierarchically defined OT for data valuation. However, while *LAVA* processes the whole dataset, *SAVA* divides the dataset into batches of data points, and carries out the OT problem computation on those batches. Moreover, our theoretical derivations on the trade-off of using entropic regularization for OT problems include refinements of prior work. We perform extensive experiments, to demonstrate that *SAVA* can scale to large datasets with millions of data points and does not trade off data valuation performance. Our Github repository is available at \url{https://github.com/skezle/sava}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new learning-agnostic data valuation approach that assigns a value to each data point in the training set based on its similarity to the validation set. They introduce SAVA, a scalable variant of the LAVA algorithm, which uses optimal transport (OT) sensitivity to value training data without directly concerning model performance efficiently. Unlike LAVA requiring the entire dataset as input, SAVA operates on batches of data points, making it has a smaller memory consumption. Thus, SAVA can make the valuation taks of larger datasets possible. The authors conduct extensive experiments showing that SAVA can effectively scale to large datasets and maintain data valuation performance across various downstream tasks.

### Strengths
[S1] An interesting approach leveraging the idea of batches to solve the memory bottleneck encountered in OT solver as optimizer in model training.

[S2] Detailed theoretical proofs and descriptions of previous work are given.

[S3] The article is well-organized and easy to read.

### Weaknesses
[W1] My biggest concern is the proof of the upper bound does not adequately explain why this proxy can work.  Detailed analysis on the upper bound of the proxy practicability should be taken.

[W2] My second concern is that the paper lacks of time complexity analysis. And SAVA in Figure 2 seems to be no better than Batch-wise LAVA. In the appendix Figure 9, why not compare Batch-wise LAVA in running time metric? 

[W3] Typos: Line 417, "Batch-wise LAVA KNN Shapley and" -> "Batch-wise LAVA, KNN Shapley, and"

Placing Table 1 in Section 2 would help to improve understanding.

### Questions
pls see W2

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops a variant of LAVA, called SAVA, for scalable data valuation. The idea is perform data valuation on batches of data instead of on the entire dataset. Extensive numerical results are presented to demonstrate SAVA's efficiency.

### Strengths
- The experimental results are convincing. The authors compared to SOTA methods for data valuation across various data corruption scenarios. The results demonstrate that SAVA is scalable to large datasets. Also, the results included a dataset of size larger than 1 million samples, in which the proposed method outperforms benchmarks. 

- The writing is good and easy to follow.

### Weaknesses
- The reviewer's biggest concern is related to novelty. Currently, SAVA seems a very natural extension of LAVA for data valuation on batches. The submission seems to be on the incremental side, unless the authors can clearly state the technical challenge when calculating on batches. 

- The choice of batch size is a key hyper-parameter in SAVA (and key difference to LAVA). The authors are suggested to include formal theoretical analysis to quantify the tradeoff in choosing batch size between memory and calculation approximation. Also, Appendix G should appear in the main text. 

- The authors are suggested to include a table comparing the complexities of LAVA and SAVA. 

- What happens if the validation dataset gets corrupted? 

- In Fig. 3, why is the performance of SAVA dropping at .4 proportion?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates of the problem of extending Optimal Transport (OT) distance-based data valuation methods for larger scale problems. The paper points out that for current methods, the quadratic overhead for expensive GPU memory constrained the scale of problems they can be applied to. Correspondingly, this paper proposes to compute the OT problem in a batch-wise fashion where the batch-wise results are aggregated via an hierarchical OT framework to produce data point valuations. This approach allows converting intractable large-scale OT problems into a series of smaller problems that can be solved efficiently. Empirical results on a variety of tasks show the proposed approach achieves competitive performance compared to original methods while being applicable to larger-scale problems.

### Strengths
The problem is well-contextualized and the motivation is clear. Structure of the paper is well balanced and the elaborations are coherent. It is straightforward for readers to understand the scope and target of the paper and the proposed technical approaches.

The proposed method is plausible, leveraging the hierarhical OT framework to aggregate results from batch-wise OT computations and achieving favorable approximation results.

Derivations are comprehensive and are paired with substantial elaborations. Empirical evaluations are diverse and abundant and the results are valid.

### Weaknesses
I am still somewhat concerned about the computation overhead for SAVA. Even it avoids directly solving large-scale OT problems and circumvents OOM issues, it now requires solving a quadratic number of OT problems between every pair of batches and aggregating their results. This could also take a significant amount of time if the number of batches are high.

Are there results on actual time comparisons for the methods in empirical studies?

The structure of the paper still has room to improve. The current layout is dense where there are many equations and lemmas interleaved with elaborations. There's an overhead for the readers to familiarize with the notations before being able to catch up with the ideas. It could be made more straightforward.

For example, the crucial Figure 1 and Algorithm 1 are not self-contained. Many of the involved notations are not straightforward and also not introduced in the captions. It still requires readers to first read through the texts and derivations to understand what is being done. Strongly suggests authors to make an effort to improve these visualizations, which could substantially improve the paper's accessibility and impact.

### Questions
Other than hierarchical OT and the proposed implementation, there are some other ideas for mitigating OT efficiency issues.

Some standard approaches include low-rank approximation to the transportation matrix C, which is often possible for practical cases. This allows representing the large matrix C with multiplication of smaller matrices and avoids directly materilizing the large matrix C and OOM issues. 

Another somewhat connected idea is to directly quantize the train and validation distributions (e.g., approximate the distributions via downsampling) to simplify the OT problem.

Hierarchical OT can also be conducted with clustering methods. For example, at the lower level, group all the samples into a number of clusters, and at the higher level, solve the OT problem between the centroids of clusters. 

It will be very interesting to see how to connect the proposed framework to these ideas and whether they may help further improving the computation complexity or accuracy.

### Soundness
3

### Presentation
3

### Contribution
3
