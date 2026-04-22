# Revisiting Multivariate Time Series Forecasting with Missing Values

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 2, 4, 6

## Abstract
Missing values are common in real-world time series, and multivariate time series forecasting with missing values (MTSF-M) has become a crucial area of research for ensuring reliable predictions. To address the challenge of missing data, current approaches have developed an imputation-then-prediction framework that uses imputation modules to fill in missing values, followed by forecasting on the imputed data. However, this framework overlooks a critical issue: there is no ground truth for the missing values, making the imputation process susceptible to errors that can degrade prediction accuracy.
In this paper, we conduct a systematic empirical study and reveal that imputation without direct supervision can corrupt the underlying data distribution and actively degrade prediction accuracy. To address this, we propose a paradigm shift that moves away from imputation and directly predicts from the partially observed time series. We introduce **C**onsistency-**R**egularized **I**nformation **B**ottleneck (CRIB), a novel framework built on the Information Bottleneck principle. CRIB combines a unified-variate attention mechanism with a consistency regularization scheme to learn robust representations that filter out noise introduced by missing values while preserving essential predictive signals. Comprehensive experiments on four real-world datasets demonstrate the effectiveness of CRIB, which predicts accurately even under high missing rates. Our code is available in [https://anonymous.4open.science/r/CRIB-F660](https://anonymous.4open.science/r/CRIB-F660).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the problem of multivariate time series forecasting with missing values and proposes CRIB, a direct prediction approach that integrates the Information Bottleneck theory, a unified-variate attention mechanism, and consistency regularization.

### Strengths
- S1: The paper conducts an empirical study that analyzes and reveals the limitations of existing “imputation-then-prediction” MTSF-M methods.
- S2: The paper conducts extensive experiments to demonstrate the effectiveness of CRIB, covering four datasets, multiple baselines, and various missing patterns.

### Weaknesses
- W1: The novelty of the paper is not clearly articulated, and the motivation behind each technical module is insufficiently explained. It appears that the method mainly combines existing components such as the Information Bottleneck, Unified-Variate Attention, and Consistency Regularization.
- W2：The paper uses an empirical study to analyze and argue that the imputation step in imputation-then-prediction methods is unreasonable. However, the details of the empirical study are not clearly explained.
- W3：The paper lacks comparisons with the latest MTSF-M baselines, such as GinAR.
GinAR: An End-To-End Multivariate Time Series Forecasting Model Suitable for Variable Missing, KDD 2024.
- W4：The paper contains some inconsistencies between the text and figures. For example, $L_{conc}$ in Figure 2(d) is not mentioned in the main text.

### Questions
- Q1: How were the distribution and correlation map in Figure 1 illustrated? What are the specific implementation details?
- Q2：Could you clarify the challenges of using direct prediction to address the MTSF-M problem, and explain what specific issues each module of CRIB tackles and what their respective technical novelties are?
- Q3：How does the paper demonstrate that consistency regularization with the IB theory can retain essential task-relevant information while filtering out irrelevant noise caused by missing values?
- Q4：How is the Imputed variant implemented in the experiment? Is there any information leakage between the training of the completion task and the prediction task?
- Q5：How are the three missing patterns, point, block, and column, simulated?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
To address the challenges posed by missing data, the authors introduce the CRIB. CRIB integrates a unified variable attention mechanism with a consistency regularization scheme. Comprehensive experiments conducted on four real-world datasets demonstrate the effectiveness of CRIB, which achieves accurate predictions even under high missing rates.

### Strengths
1. The article is well-written.
2. Prediction with missing values ​​is a significant issue.
3. The existing experiments demonstrate the model's excellent performance.

### Weaknesses
1. The motivation has already been explored. To my knowledge, the limitations of the "impute-then-predict" paradigm have been repeatedly discussed [1-3]. Additionally, CSDI also addresses the lack of authenticity in missing values during training and proposes a self-supervised learning strategy. Therefore, the authors should present a more compelling motivation. Furthermore, some imputation works that do not explicitly emphasize their applicability to prediction tasks (e.g., TimeNet, TimeMixer++) and some prediction works capable of directly handling missing values should be considered as competitors in this paper and discussed and compared in detail. Otherwise, this approach may lead to an overestimation of the paper's contributions.

2. The "impute-then-predict" approach is not the mainstream framework adhered to by existing methods. Some datasets naturally contain missing values, and many prediction works do not perform an additional imputation step—they directly make predictions. The "impute-then-predict" paradigm generally involves simulating missing data through random masking, which is more akin to a self-supervised learning approach.

3. The paper selects a relatively weak-performing model (DLinear) (which only consists of a few fully connected layers) and a heterogeneous model (TimesNet) to construct the "impute-then-predict" paradigm and uses them as the primary targets of criticism. The rationale for this motivation is insufficient. This experimental setup appears more like a "fragile" control group constructed to highlight the advantages of the proposed method, rather than a fair evaluation of the state-of-the-art or most commonly used paradigms in the real world. A more convincing analysis should directly compare with specialized, powerful imputation models (e.g., CSDI, ImputeFormer) or advanced prediction models capable of directly handling missing values (e.g., appropriate variants of D2STGNN, TimesNet), and delve into the potential limitations of these methods.

4. Furthermore, the authors' first contribution claims: "We perform a systematic empirical analysis of the dominant imputation-then-prediction paradigm for MTSF-M. We reveal that, guided only by a prediction objective, imputation modules can corrupt the observed data distribution and degrade prediction performance" seems exaggerated. The authors use a toy example to support their research, which is not systematic or concrete. Reliable findings should be based on either theoretical foundations or extensive experiments.

5. What challenges would other models face if performing direct prediction? Regarding these challenges, what specific strategies has your model implemented?

6. Although the authors introduce the information bottleneck theory, what is the utility of the lower bound proven in Section 3.4.2? In my view, if it's just an isolated value, it's difficult to grasp its meaning. For instance, the authors should introduce the lower/upper bounds of other methods for comparison to illustrate the significance of this bound.

7. Several imputation models are missing, such as PriSTI, CSDI, and ImputeFormer. Additionally, some spatiotemporal forecasting models are missing, e.g., D2STGNN,STID,PatchSTG. Some time series models are also missing, such as DUET, CycleNet, and TimesNet. Some commonly used time series data sets, such as Weather, Traffic, etc., need to be considered.

8. Furthermore, I recommend denormalizing all prediction results; this is standard practice for PeMS and Metr-LA datasets.

9. I suggest adding MAPE as a third metric.

10. The work appears to be a combination of mature components. The authors' temporal analysis model is a standard Transformer, and the theory is derived from the well-established mutual information/variational inference theory. Therefore, it seems more like an engineering effort of combination rather than an original contribution.

11. Did the authors evaluate the performance of an ablated variant without the Consistency Regularization term? This is necessary to demonstrate that the performance gain indeed stems from this module.

12. The authors' loss function involves three hyperparameters, which could clearly lead to optimization difficulties. Therefore, a detailed sensitivity analysis is required. Figure 4 alone is insufficient.

13. The code cannot be run directly. It would be better if you can add more detailed instructions for using the code.

Reference:

[1] Tashiro, Yusuke, et al. "Csdi: Conditional score-based diffusion models for probabilistic time series imputation." Advances in neural information processing systems 34 (2021): 24804-24816.

[2] Peng, J., Yang, M., Zhang, Q., & Li, X. (2025). S4M: S4 for multivariate time series forecasting with Missing values. arXiv preprint arXiv:2503.00900.

[3] Chen, X., Li, X., Liu, B., & Li, Z. (2023). Biased temporal convolution graph network for time series forecasting with missing values. In The Twelfth International Conference on Learning Representations.

### Questions
W1-W13.

### Soundness
2

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
3

### Summary
This paper presents CRIB, a novel method for Multivariate Time Series Forecasting with Missing Values, demonstrating state-of-the-art performance across four diverse datasets in an extensive experimental evaluation. The work addresses a critically important real-world challenge. However, several significant limitations temper these strengths. The practical relevance is undermined as experiments use artificially induced missing data on complete, highly periodic datasets, failing to test on real-world, non-stationary, or high-dimensional data. The problem setup's generalizability is weakened by an imputation strategy that ignores realistic missing mechanisms (e.g., MNAR). Furthermore, the core innovation of unified-univariate attention lacks theoretical grounding, and claims of superiority over other models remain unconvincing without deeper analysis into why they fail. Consequently, while CRIB shows promising results, its validity and applicability to genuine real-world scenarios are questionable.

### Strengths
S1. The paper addresses the increasingly important problem of Multivariate Time Series Forecasting with Missing values, a critical challenge in real-world applications such as healthcare, environmental monitoring, and industrial IoT, where data completeness cannot be assumed.

S2. The experimental section is extensive and methodologically sound, covering four diverse datasets

S3. The proposed method, CRIB, achieves state-of-the-art performance across all datasets and metrics

### Weaknesses
W1. The paper claims to address real-world scenarios where ground-truth values are unavailable. However, the two-stage imputation strategy—training TimesNet on 10% missing data and applying it to higher missing rates—does not reflect real-world missing mechanisms (e.g., MNAR due to sensor failure or human behavior). The absence of discussion on missing mechanisms (MCAR/MAR/MNAR) weakens the practical motivation and limits the generalizability of the problem setup.

W2. The paper asserts superiority over existing methods but fails to explain why models like SAITS or BiTGraph degrade under high missing rates. A deeper analysis—e.g., error decomposition, attention pattern visualization, or gradient sensitivity—would strengthen the motivation. Without such analysis, the claim that "existing methods fail" remains superficial and unconvincing.

W3. The unified-univariate attention is a core innovation, yet the paper offers no theoretical analysis of its properties—e.g., representational capacity, gradient flow, or robustness to missing data. Why is this structure more suitable for MTSF-M than standard self-attention or cross-variate attention? Without such grounding, the method appears heuristic rather than principled.

W4. Experiments are conducted on only four datasets: PEMS-BAY, Metr-LA, ETTh1, and Electricity. These are all highly periodic, low-noise, relatively low-dimensional and . The lack of testing on non-stationary, high-dimensional (e.g., >1000 variates), or irregularly sampled datasets limits the generalizability of the results and raises concerns about overfitting to specific data characteristics

W5. The experimental evaluation is conducted on only four datasets. While these are commonly used in time series forecasting, they share highly similar characteristics: strong periodicity, low noise, regular sampling, and relatively low dimensionality (tens to hundreds of variables). Crucially, none of these datasets contain real missing values. Instead, the authors artificially inject missing data under controlled patterns (point, block, column), which significantly weakens the practical relevance of the study.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes that imputing missing values may be inaccurate, which can affect the accuracy of time simulation and thereby harm the precision of future predictions. To mitigate the errors brought by imputation, the paper further proposes a novel framework called CRIB, constructed based on the information bottleneck principle. CRIB combines a unified variable attention mechanism with a consistency regularization scheme to learn robust representations, filter out noise introduced by missing values, and meanwhile preserve the fundamental predictive signals. The experimental results of the paper show that the proposed method outperforms existing classic baselines.

### Strengths
This paper proposes that imputing missing values may be inaccurate, which can affect the accuracy of time simulation and thereby harm the precision of future predictions. To mitigate the errors brought by imputation, the paper further proposes a novel framework called CRIB, constructed based on the information bottleneck principle. CRIB combines a unified variable attention mechanism with a consistency regularization scheme to learn robust representations, filter out noise introduced by missing values, and meanwhile preserve the fundamental predictive signals. The experimental results of the paper show that the proposed method outperforms existing classic baselines.

### Weaknesses
1. The experiments in this paper are unreasonable. From the source code, this paper fills the missing value part with 0, which is actually imputing the data with a value of 0, severely disrupting the original temporal patterns. In the subsequent forward and backward propagation processes, the filled 0 values are involved in the calculations. This will seriously affect the model's optimization and lead to training failure. This may also be why the baselines in Figure 1 perform poorly, as they are trained under conditions where the data patterns are severely disrupted (real values are randomly replaced with 0). It may be that even simple non-model cubic function interpolation would perform better than the models trained in Figure 1. Missing value parts should not be involved in calculations. They should be handled using masks during the network computation process, or using a process like neural CDE for processing, instead of filling the missing parts with zero at the input stage.

2. Due to the unreasonable handling of missing values, many of the modules proposed in the paper are unreliable. For example, the KL divergence under IB theory and the consistency constraints under noise are closely related to setting missing values to 0 for participation in forward and backward propagation, but this is meaningless. For example, random noise will replace the 0 value positions with more meaningful values, and the effectiveness may be because the 0 values are replaced.

3. Although the logic of using INFORMATION BOTTLENECK GUIDANCE in the paper is reasonable, the approach is confusing. Calculating the Gaussian posterior based on features and making it close to the standard Gaussian prior with the KL divergence seems to have no supervisory information. Why can this minimize mutual information and filter out irrelevant information?

4. Some parts of the paper are vaguely described. W/o IB means removing the compactness and informativeness guidance of IB. The former is Eq. 9, but what is the latter? The two are mixed in the ablation study, and the effect of removing Eq. 9 alone is not shown.

5. The paper lacks key baselines. For example, baselines for handling irregular and missing data, such as Neural ODE or CDE.

6. The paper flattens all variable patches to calculate attention, which is not a very efficient or reasonable approach. When the number of variables is large, e.g., a traffic dataset, this method will become infeasible due to the overly long patch sequence.

7. The experiments in the paper are not solid. The main experiments are only conducted on four datasets, and the ablation studies are only performed on one dataset.

### Questions
Although the logic of using INFORMATION BOTTLENECK GUIDANCE in the paper is reasonable, the approach is confusing. Calculating the Gaussian posterior based on features and making it close to the standard Gaussian prior with the KL divergence seems to have no supervisory information. Why can this minimize mutual information and filter out irrelevant information?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on the critical task of Multivariate Time Series Forecasting with Missing Values (MTSF-M), where missing values are prevalent in real-world scenarios. It first points out the flaw of the current mainstream "imputation-then-prediction" framework: since there is no ground truth for missing values, unsupervised imputation is prone to errors, which corrupts the underlying data distribution and degrades forecasting accuracy—a conclusion validated by the authors’ systematic empirical study. To address this issue, the paper proposes a paradigm shift: abandoning independent imputation and directly forecasting from partially observed time series.

### Strengths
The paper identifies a critical limitation of the mainstream "imputation-then-prediction" framework in MTSF-M—unsupervised imputation’s lack of ground truth leading to data distribution corruption and forecasting accuracy degradation—and validates this via systematic empirical studies, which provides a clear and necessary critique of existing methods. Additionally, the proposed paradigm shift, direct forecasting from partially observed data, and the CRIB framework  demonstrate theoretical innovation.

### Weaknesses
The paper’s central claim—shifting from “imputation-then-prediction” to direct forecasting via an IB-based framework—is conceptually relevant but lacks sufficient novelty to meet ICLR standards. The critique of imputation-induced distribution corruption is not fully original. Prior works have already questioned the imputation paradigm and proposed direct prediction methods, yet these are not adequately discussed or contrasted. The proposed “CRIB” framework combines two existing components—Information Bottleneck (IB) and consistency regularization—without demonstrating a non-trivial, innovative integration. The paper fails to explain how CRIB’s design solves unique challenges that prior IB-based or consistency-regularized models cannot address. The paper only compares against “imputation-then-prediction” methods but omits state-of-the-art direct forecasting baselines This one-sided comparison overstates CRIB’s performance.

### Questions
The paper only claims CRIB is effective but does not compare it to SOTA MTSF-M models. Could you add a head-to-head comparison with recent methods?
The paper references "four real-world datasets" but provides no context to assess generalizability. Please list the dataset names, domains, time series length, and number of variables for each dataset?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new paradigm for time-series forecasting with missing data based on the Information Bottleneck Principle. They argue that current forecasting with missing data tends to use imputation modules to recover the missing data, then do the forecasting. This is claimed to be flawed because of the lack of ground truth for missing data, which can corrupt the observed data and hurt the performance. As an alternative approach, they (i) propose an architecture that handles the sparse nature of the observed data through patching embedding and unified-variate attention, and (ii) include the consistency regularization loss additionally to the usual reconstruction and compactness losses, along with data augmentation to learn invariance to missing patterns. They show that their scheme provides strong MAE and MSE performance on four datasets against twelve baselines with four missing rates settings.

### Strengths
The paper achieves strong performance against the baselines (Table 1), and multiple ablations that show the importance of the different modules (Table 2 and Figure 4), including the Consistency loss and the Unified-Variate Attention. In particular, Figure 4 (a) shows the importance of (i) IB (ii) Consistency loss, and (iii) Uni-Attn, and Figure 4 (b) shows the robustness to hyperparameters. Also, Table 2 shows that CRST-IB can perform significantly better than w/o for well-chosen consistency regularization weights.

### Weaknesses
There is too much information in Figure 1: it is hard to read. You should simplify it and put your method on the right. It would also be nice to add more qualitative comparison with the competitors. Although it is said in the Related Work section, it should be made clearer in the paper that Compactness and Informativeness losses already exist and are not part of the novelty. This needs to be clearer both in 3.4.1, 3.4.2, and precise references of previous work in the proofs in Appendix B. 

Also, the justification of the consistency loss, e.g with "The core intuition is that the model’s prediction should be invariant to the missingness." is not clear. Can you better explain that?

Issues/typos:

L.42: $\\beta \in \\mathbb{R}^{+}_{*}$

Eq.3: Shouldn't it be $\sin(t/10000^{2m/P})$ instead (and same for cos(...)) ? Otherwise, the embedding doesn’t depend on $m$ when the parity is fixed.

L.277-282: 

Sign issue in Eq.10

$\\mathbb{E}\_{p(z, y)}[\log q_\\theta(y \mid z)]= \textcolor{red}{-} \mathbb{E}\_{p(z, y)}[\frac{1}{2 \\sigma^2}\||y-\\hat{y}\||^2+\frac{T}{2} \\log \left(2 \\pi \\sigma^2\right)]$

Figures 3 and Figure 4a: It should be better to have consistency in color for the method (e.g., yellow).

### Questions
See weaknesses, also:

What are the main limitations of the method ? Is that the amount of hyperparameters ? How did you choose these ($\\alpha, \\beta, \\gamma$, 10% in the additional random masking, variance of the added Gassian noise) and what are the chosen hyperparams for the Figure 4a ? Limitations should be acknowledged in a separated section.

Are the competitors comparable in terms of computational cost ? Did you retrain the baselines from scratch ?

It would be nice to add the model trained with clean data as well (missing rate = 0) to get a bound on the achievable performance.

### Soundness
2

### Presentation
1

### Contribution
2
