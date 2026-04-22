# A Fast Kernel-based Conditional Independence Test with Application to Causal Discovery

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 4, 4, 6, 6

## Abstract
Kernel-based conditional independence (KCI) testing is a powerful nonparametric method commonly employed in causal discovery tasks. Despite its flexibility and statistical reliability, cubic computational complexity limits its application to large datasets. To address this computational bottleneck, we propose FastKCI, a scalable and parallelizable kernel-based conditional independence test that utilizes a mixture-of-experts approach inspired by embarrassingly parallel inference techniques for Gaussian processes. By partitioning the dataset based on a Gaussian mixture model over the conditioning variables, FastKCI conducts local KCI tests in parallel, aggregating the results using an importance-weighted sampling scheme. Experiments on synthetic datasets and benchmarks on real-world production data validate that FastKCI maintains the statistical power of the original KCI test while achieving substantial computational speedups. FastKCI thus represents a practical and efficient solution for conditional independence testing in causal inference on large-scale data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the cubic-time computational cost of the Kernel-based Conditional Independence Test (KCIT). The proposed method, FastKCI, introduces a scalable and embarrassingly parallel framework that partitions the data based on a Gaussian mixture model over the conditioning variable $Z$. For each of $J$ independently sampled partitions, the method performs local KCI tests within $V$ clusters and aggregated the resulting statistics via importance weighting. Under mild assumptions, the approach preserves the null distribution of the original KCIT while reducing the effective runtime to approximately $O(Jn^3/V^2)$. Empirical results on synthetic and causal discovery benchmarks show that FastKCI achieves comparable test accuracy while substantially improving computational efficiency.

### Strengths
1.	Significance: The paper targets the computational bottleneck of KCIT, which is a crucial component in constraint-based causal discovery. Scaling KCI to large dataset is an important and timely problem, and the paper’s framework offers a novel and clear way to parallelize and accelerate the test.
2.	The method is well-aligned with existing theory and preserves statistical validity. FastKCI maintains the null distribution of the original KCIT under mild conditions, and the derivation is clearly explained. 
3.	Quality: The experiments are relatively thorough. The empirical evaluation demonstrates that the method achieves similar Type-I and Type-II error performance to KCI, while reducing the runtime. The scenarios considered are generally appropriate, and the results are consistent with the claims.

### Weaknesses
1.	Some modeling and notation choices are not explained in sufficient detail. For example, in Assumption 1, the Normal-Inverse-Wishart prior is introduced, but there is no introduction to the hyperparameters and no guidance on how its hyperparameters are selected in practice or how sensitive the method is to these choices. In addition, some presentation issues (e.g., the placement of Table 2 below the footnote) hinder readability.
2.	The paper provides a discussion on how to choose the variable $J$ and also gives some recommendations for choosing $V$. However, it might be practically more useful to choose $V$ according to the sample size, e.g., making $V$ a function of the sample size $n$. It would be helpful if the authors could clarify whether this is feasible.
3.	Not enough assessment with respect to the performance comparison between other baselines and FastKCIT (see question part below).

### Questions
1.	In the related work section, several approaches for accelerating KCIT or designing fast conditional independence tests are mentioned. However, the experimental evaluation only includes KCIT and RCIT in a subset of settings. Given that the main claim of the paper is scalability while preserving statistical performance, a broader comparison against these established fast CI tests would be necessary to support the empirical contribution. I suggest including more methods to more fully substantiate the claimed advantages.
2.	More simulation results appear to be based on data generated from Gaussian mixture distributions of $Z$, which align with the modeling assumption. To better understand robustness, could the authors provide results on data where $Z$ is not well modeled by Gaussian mixtures (beyond the production dataset), or discuss expected behavior in such cases?
3.	For the comparison with RCIT, could the authors provide the detailed data-generating process used in Section 5? Since RCIT often performs well in practice, a clearer description of the experimental setup would clarify under which conditions FastKCI has an advantage.
4.	In Table 4, FastKCI shows strong precision performance (with a loss in recall). Could the authors provide intuition or explanation for this behavior, especially given that the performance differences in earlier synthetic experiments were relatively small?

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
This paper proposes FastKCI, a scalable and parallelizable variant of the Kernel-based Conditional Independence (KCI) test designed to overcome the computational bottleneck of KCI without sacrificing its statistical power. FastKCI's approach is based on a "mixture-of-experts" model and parallel inference techniques. The dataset is partitioned into $V$ smaller subsets, and the partitioning is guided by fitting a Gaussian mixture model (GMM) to the conditioning variables $Z$. A standard KCI test is then conducted on each of these smaller partitions independently and in parallel. Finally, the results from all local tests are combined using an "importance-weighted sampling scheme" to produce the final global test statistic and p-value. The method is validated on synthetic datasets and applied to real-world data for causal discovery. The results confirm that FastKCI achieves comparable accuracy to KCI in causal discovery tasks but in a fraction of the time.

### Strengths
1. The paper has a clear motivation and is structured. 
2. CI tests lie in the computational bottleneck in constraint-based causal discovery.

### Weaknesses
1. The reliability of FastKCI heavily depends on the Gaussian Mixture Model (GMM) assumption. If the true distribution of $Z$ is not well approximated by a GMM, the performance of FastKCI may significantly degrade. 
2. The current simulation experiments are conducted only on data generated from Gaussian mixture models. Additional experiments on data that violate the GMM assumption are necessary to assess the robustness of FastKCI under model misspecification. 
3. As a practical improvement, it would be useful to include a preliminary goodness-of-fit test. For instance, if the fitted GMM poorly represents 
$Z$, the algorithm could either alert the user or automatically adjust the number of partition samples $J$ to mitigate the mismatch. 
4. FastKCI introduces two hyperparameters—$V$ (the number of mixture components) and $J$ (the number of partition sampling rounds)—yet the paper provides insufficient guidance on how to select these parameters in practice. 
5. From a methodological perspective, the paper’s novelty appears somewhat limited, as the main contribution lies in transferring an existing idea (embarrassingly parallel inference) to the conditional independence testing setting. Nevertheless, the adaptation is technically sound and potentially useful in practice. 
6. Despite the claimed efficiency, FastKCI still exhibits cubic computational complexity with respect to the sample size $n$.

### Questions
1. Would it be possible to design an optimization criterion or data-driven procedure to automatically determine the optimal values of $V$ and $J$? 
2. What are the key advantages of adopting embarrassingly parallel inference techniques compared to kernel acceleration approaches such as incomplete Cholesky decomposition, random Fourier features, or the Nyström approximation?
3. Could the authors clarify how the p-value is computed in FastKCI? For example, is it based on an asymptotic null distribution, permutation testing, or another resampling approach?

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
4

### Summary
This paper proposes an accelerated algorithm for the kernel-based conditional independence test (KCI), termed FastKCI. The FastKCI method assumes that the conditioning variable $Z$ follows a Gaussian Mixture Model (GMM), and partitions the original dataset of $n$ samples into $V$ groups according to the $V$ components of the GMM. This reduces the computational complexity from $O(n^3)$ to $O(n^3 / V^2)$. Experimental results demonstrate that FastKCI significantly improves computational efficiency.

### Strengths
FastKCI substantially reduces computational time in most cases without significantly compromising the accuracy of the CITs.

### Weaknesses
1. The paper merely combines embarrassingly parallel inference with KCI in a straightforward manner, leading to limited novelty.  
2. Although FastKCI achieves faster computation, the experimental results show that the accuracy of the CITs decreases in certain cases. Moreover, the paper only provides a coarse theoretical analysis for the asymptotic behavior of FastKCI as $n \to \infty$, which fails to explain the observed loss in accuracy.  
3. The experiments compare FastKCI only with other KCI-based baselines, and the experimental settings focus mainly on low-dimensional conditioning variables $Z$ (e.g., Sections 5.1, 5.3). In contrast, recent CIT methods can efficiently handle high-dimensional $Z$ [GCIT, DGCIT, NNLSCIT, …]. This raises an important question: can the embarrassingly parallel inference framework be extended to aggregate the test statistics of other CIT methods?  
4. Several parts of the paper are not clearly articulated; see the Questions section for details.

### Questions
1. How are the means, variances, and weights of the different components in the Gaussian Mixture Model estimated? Are they obtained using the EM algorithm? This should be described more explicitly in the paper.  
2. I suggest that the authors carefully check the relationship between $X$ and $Y$ across the $V$ different subsets. That’s because the CI relationships in different clusters may change, especially in real data. If Z indeed follows the GMM assumption, do $X$ and $Y$ maintain a consistent conditional independence (CI) relationship across components? Conversely, if $Z$ does not follow the GMM assumption—meaning that the partitioning of samples into $V$ subsets may be not well—how would the CI relationship between $X$ and $Y$ change?  
3. In Line 230, what is the exact form of the distribution $P(X^{(j)}, Y^{(j)} | U^{(j)})$ that $l^{(j,v)}$ follows? This requires a more detailed explanation.  
4. The experiments shown in Appendix A.1 indicate that the accuracy of FastKCI seems to depend heavily on the correct estimation of the number of components $V$. If we ignore computational speed and focus solely on the accuracy of the CIT, how can one determine an appropriate value of $V$?  
5. I would like to know how \textit{Precision} and \textit{Recall} are defined in Tables 2 and 9–11. Specifically, how is the confusion matrix constructed, and how do Precision and Recall relate to Type I and Type II errors?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is related to causal discovery and proposes a kernel-based Conditional Independence Test (FastKCI), which uses a divide-and-aggregate strategy to reduce the typical cubic numerical complexity in the number of samples. The approach breaks the full kernel into V smaller parts (data partitions). Several numerical experiments are performed to illustrate the performance of the ensemble approach. Results for different baseline setups look promising. Type I error and statistical power remain competitive while saving computational time.

### Strengths
S1 The considered problem is interesting and relevant.
S2 The paper is technically sound and overall well-written.
S3 To reduce complexity by 1/V^2 is impressive.
S3 The evaluation and comparison against baselines is convincing.

### Weaknesses
W1 The scalability regarding the number of involved variables remains somewhat unclear.
W2 Limitations of the paper’s methods could be better discussed.

### Questions
Does it make a difference how the data is partitioned into the V subsets?

Regarding Table 3: Does the approach scale in |Z|? Is the approach still fast if the number of involved variables is 100 or 1000?

How should the hyper parameter V be automatically chosen in general applications? Is the trade-off between accuracy and speed plannable?

The figures and tables are too small. I understand that there is a lack of space but font sizes in figures and tables should be as in the text.

Under which conditions the approach performs not good (e.g. regarding the data partitioning when highly correlated clusters are chosen/realized)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a scalable, kernel-based conditional independence testing approach that first partitions the data into subsets using a Gaussian mixture model, then performs parallel conditional independence tests, and finally aggregates the results using an importance-weighted sampling scheme.

### Strengths
-	The paper presents a strong motivation and a practical approach for improving the efficiency of conditional independence tests, and consequently, constraint-based causal discovery.
-	The paper is well-written and clearly presented.

### Weaknesses
-	Missing relative references. Please see the question 1 and 2 in Questions section.
-	This paper provides limited evaluation of generality: missing real-world applications, limited settings on synthetic data. Please see questions 5, 6, and 7 in Questions section.

### Questions
1.	I believe [1] also partitions the data into subsets, applies parallel kernel tests, and aggregates the results for efficient conditional independence testing. This work is highly relevant and should be discussed in the paper.
> [1] Guan, Zhengkang, and Kun Kuang. "Efficient Ensemble Conditional Independence Test Framework for Causal Discovery." arXiv preprint arXiv:2509.21021 (2025).
2.	How does the computational complexity of fastKCI compare to [1]? [1] claims linear complexity in sample size \(n\). For fastKCI to achieve similar complexity, it requires \(V = n\) components, which may be impractically large.
3.	What is the purpose of assuming a Normal-Inverse-Wishart prior on \(Z\)?
4.	Does the accuracy of the aggregated measurements depend on the reliability of individual measurements? Additionally, will errors in learning the mixture Gaussian distribution affect the accuracy of subsequent steps?
5.	Why do the authors assume nonlinear causal functions? It would also be informative to report results for linear causal functions.
6.	Sample size may not be the dominant factor affecting causal discovery efficiency. The main challenge for constraint-based approaches lies in the accuracy and scalability of conditional independence tests when the conditioning set is high-dimensional. How does the proposed approach perform under such scenarios?
7.	Although the empirical results on semi-synthetic data are promising, it would be valuable to evaluate performance on real-world datasets, such as Sachs et al. (2005).

### Soundness
3

### Presentation
3

### Contribution
3
