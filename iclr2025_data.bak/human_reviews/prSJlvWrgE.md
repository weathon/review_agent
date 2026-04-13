## Human Reviewer 1

### Summary
This work proposes a new framework to handle the challenging concept drift detection and adaptation of time-series data. It first implements kernel representation learning to obtain the concepts, and then exploits the inter-dependence of different time-series for future time-series prediction. Moreover, the proposed method can be easily combined with deep learning models for performance enhancement.

### Strengths
1. A new powerful time-series data analysis framework has been proposed. It can learn concepts, detect drifts, and adapt to the drifts by predicting future time series. The basic idea is sound.

2. The diagrams can intuitively demonstrate the principles of the proposed method.

3. Comprehensive experimental evaluation.

4. Source code has been opened.

### Weaknesses
1. The idea is straightforward, derived from the stream of kernel trick-based representation learning, where the non-trivial selection of kernel functions may limit the application of the proposed method.

2. The efficiency of the proposed method has not been discussed by providing complexity analysis or execution time evaluation.

3. The work focuses more on drift adaptation, while the drift detection ability and processes of the proposed method have not been well discussed and evaluated.

### Questions
1. Have considered techniques like multiple kernel learning, adaptive kernel selection, and automatic selection of the kernel parameters?

2. What is the time and space complexity of the proposed method? Have you compared the execution time of the proposed one with the SOTAs on large-scale datasets?

3. How about the Type-I and Type-II errors of the proposed method in concept drift detection?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper discusses an interesting mechanism for forecasting over co-evolving time-series by not only correcting for future drifts, but also for detecting new concepts. Here, concepts are defined as clusters of profile patterns across similar subseries within a specific
window. Each subseries is represented as aa linear combination of other series, and the matrix that captures this self-representation is learned by employing the kernel trick. The authors try to introduce numerous mechanisms to capture overall dynamics of concepts/ cluster evolution including a new kernel representation learning strategy, a nonconvex optimization strategy, and a drift detection strategy. They also show how these can be integrated into an auto-encoder. The empirical results demonstrated in the paper clearly show that evolving concepts are well captured by the model compared to SOTA.

### Strengths
1) The drift adaptation methology seems very elegant.
2) The theoretical underpinning of the proposed methodology is well explained.
3) Details given in the appendix is very useful to understand the contributions.

### Weaknesses
(1) The forecasting performance shown in Table 1 does indicate that D2M generally has lower RMSE in comparison to other
SOTA methods. However, those values to close to its competitors. It is hard to understand the effectiveness of the approach without any variance or significance measures such as confidence intervals for the RMSE values, P-values from statistical significance tests comparing Drift2Matrix to competing methods and standard deviations of RMSE across multiple runs. These additions would allow for a more rigorous comparison between Drift2Matrix and competing methods.

(2) While the details in the Appendix provide key insights into the working of the proposed method, it is important to summarize key details within the paper for it to be fully contained. A reader should not be forced to rely on the Appendix to understand the setup. For example, solving the non-linear optimization uses a specialized method as indicated in the Appendix. However, there is no mention of the said ADM strategy in the main paper.  A brief overview of the ADM strategy for solving the non-linear optimization problem in the methodology section, key aspects of the kernel representation learning strategy and a concise explanation of the drift detection strategy, would be useful to ensure that readers can understand the core methodology without relying heavily on the Appendix.

This also raises another important issue. 
(3) The authors seem to introduce too many methodologies within the paper while validating only some of them. Including the following might help improve the understanding better: Information about the established nature or novelty of the nonconvex optimization methodology used, including references to relevant literature if it's a well-established method, and ablation studies or comparative analyses for any other introduced methodologies that currently lack validation would be important to highlight in the paper. 
This would help address the concerns about the numerous methodologies introduced in the paper and provide a more comprehensive validation of the approach.

### Questions
(1) The proposed methodology seems to work well when there is a large number of co-evolving time-series. Does the number and length of such co-evolving time-series affect performance? An ablation study on these parameters would be pertinent.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper focuses on the research problem of concept drift adaptation in evolving time series, and a novel framework called Drift2Matrix that leverages kernel-induced self-representation has been proposed. The whole paper is well organized with detailed formulation and theoretical analysis. Sufficient experiments have been given for model evaluation.

### Strengths
1. The topic of concept drift learning in co-evolving time series is interesting, and the proposed method based on kernel-induced self-representation shows its novelty and ideal performance. 
2. The theoretical analysis of concept drift adaptation in this paper is well expressed and proofed.

### Weaknesses
1. The problem definition needs further strengthened, the scenario of concept drift in the co-evolving time series should be defined in detail.

### Questions
1. Is the learning mode in this paper a prequential test-then-train? If not, please given an explanation of the learning mode in this paper.
2. The experiment results of the proposed method seem not the best on some datasets, please give a detailed explanation.
3. Some benchmark methods used in this paper are proposed before 2020, I suggest continuing to add benchmark methods in the past three years.
4. Although the parameter setting has been given, the parameter sensitivity analysis is still required.

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

## Human Reviewer 4

### Summary
This paper focuses on the prediction of co-evolving time series under the occurrence of concept drift. To achieve accurate forecasting results, this paper proposes a kernel-based learning mechanism to learn representations that can capture the concepts among the co-evolving time series and thus can make accurate predictions by the learned representations.

### Strengths
This paper reviews the concept drift issue under multivariate time series forecasting and proposes a new method using kernel-based representations to identify and predict potential concepts, aka, patterns and make time series forecasting results that consider the adaptation of concept drift.

The idea has been clearly presented, with reasonable motivation. A comprehensive comparison has been conducted to show its effectiveness in forecasting over a variety of time series forecasting tasks.

### Weaknesses
The presented method is less novel considering [r1] has been the first study that address concept drift in time series forecasting.

The solution is less creative, which weakens its contribution to both multi-variant time series forecasting and concept drift adaptation domain. Clustering the data into separated concepts for concept drift adaptation is less creative. For example, the techniques used in r1, can also be considered separating the patterns by an ensembling mechanism. For me, it is less motivated and less clear here why kernel-based representation is a more advanced tool rather than ensembling. If that's the way to make a technical contribution, any new techniques which has the functionality of clustering can take a role.

There are concerns about some technical details which will be specified in the question section.

[r1] Q Wen, W Chen, L Sun, Z Zhang, L Wang, R Jin, T Tan. OneNet: Enhancing time series forecasting models under concept drift by online ensembling. NeurIPS 2024.

### Questions
1. From a high level understanding, this paper claims that existing methods consider "most multivariate models to define the concept as a collective behaviour of streaming data, falling short in their ability to capture the dynamics of individual series and their interactions". However, OneNet has considered the multivariate data through cross-time and cross-variable branches. Therefore, I didn't see a significant gap here.

2. Based on 1, the research question proposed here "Can we identify underlying concepts from co-evolving time series and leverage their nonlinear relationships to predict concepts that have not appeared in a single series?" is less convincing. Based on the experimental analysis, the aim of proposing Drift2Matrix is to increase the forecasting accuracy. However, why identifying concepts can be a more advanced way to achieve that aim than existing studies where these concepts have been considered in their model in a different way.

3. In addition, it is less rigorous to "predict concepts that have not appeared in a single series". Here, concepts are assumed to be distinct concepts, which means they are not overlapped with each other. According to the definition of concept drift, p(t) \neq p(t-1), the future concepts should not be predictable if concepts are distinct because Event of  p(t) \neq p(t-1) and the Event of p(t+1) \neq p(t) will be independent with each other. I think here the authors have mixed up temporal dependency among x(t) with dependency among concepts. That's why I don't think the THEORETICAL ANALYSIS section is the "genuine" theorem support for the proposed method.

4. Based on all the above, the experimental result looks fine but it does not exactly support the methodology from my understanding.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
5