# ENHANCING MULTIVARIATE TIME SERIES FORECAST- ING WITH MUTUAL INFORMATION-DRIVEN CROSS- VARIABLE AND TEMPORAL MODELING

- Decision: Reject
- Scores: 6, 6, 6

## Abstract
Recent researches have showcased the significant effectiveness of deep learning techniques for multivariate time series forecasting (MTSF). Broadly speaking, these techniques are bifurcated into two categories: Channel-independence and Channel-mixing approaches. While Channel-independence models have generally demonstrated superior outcomes, Channel-mixing methods, especially
when dealing with time series that display inter-variable correlations, theoretically promise enhanced performance by incorporating the correlation between variables. However, we contend that the unnecessary integration of information through Channel-mixing can curtail the potential enhancement in MTSF model performance. To substantiate this claim, we introduce the Cross-variable Decorrelation Aware feature Modeling (CDAM) for Channel-mixing approaches. This approach is geared toward reducing superfluous information by minimizing
the mutual information between the latent representation of a single univariate sequence and its accompanying multivariate sequence input. Concurrently, it optimizes the joint mutual information shared between the latent representation, its univariate input, and the associated univariate forecast series. Notably, prevailing techniques directly project future series using a single-step forecaster, sidelining the temporal correlation that might exist across varying timesteps in the target series. Addressing this gap, we introduce the Temporal correlation Aware Modeling (TAM). This strategy maximizes the mutual information between adjacent sub-sequences of both the forecasted and target series. By synergizing CDAM and TAM, we sculpt a pioneering framework for MTSF, named as InfoTime. Comprehensive experimental analysis have demonstrated the capability of InfoTime to consistently outpace existing models, encompassing even those considered state-of-the-art.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a framework for multivariate time series forecasting called InfoTime, which combines the Cross-variable Decorrelation Aware feature Modeling (CDAM) and the Temporal correlation Aware Modeling (TAM) approaches. The CDAM module filters unnecessary cross-variable correlations by minimizing the mutual information between the latent representation of a single univariate sequence and other series, and the TAM module performs auto-regressive forecasting and captures the temporal correlations across varied timesteps. The InfoTime framework outperforms existing models and can discern cross-variable dependencies while avoiding the inclusion of unnecessary information.

### Strengths
1. This paper provides detailed formula derivation to prove the model design.
2. The framework achieves promising results that outperform existing models and improves performance for both channel-mixing and channel-independence models.

### Weaknesses
1. The experiments are not extensive. It seems there are some cherry-pickings in particular configurations. How do the performance of ETTh2 and PEMS in Table 1? Some experiments (Table 2,3,4 and Figure 4) only use the results on partial datasets and baseline models. The prediction visualization of Figure 1 (comparing Informer and PatchTST) seems quite different from the reported results in the PatchTST paper.  I hope the authors can elaborate more on it.
2. This paper essentially needs further polishing. There are many screenshots as figures, uncompiled citations (in related works), and confusing notations (e.g. unstated $I^i$, confused use of $X_i$ and $X^i$, unreasonable usage of the superscript: $X^o$).
3. The model benchmark tested in the proposed framework can be too few to be representative. The latest models in different categories are encouraged to be included, such as Transformers (Autoformer), linear models (TiDE, TSMixer), and TCN-based models (TimesNet). Datasets with more variables (such as Solar-Energy) are also recommended since the method essentially addresses the cross-variable issue.
4. The term "single-step forecaster" can be confusing. I think it in fact outcomes the "multi-step" prediction. And it seems $\lambda=1$ is favored in most cases (even though experiments in Section 4.3 are not extensive). How to support the motivation of Equation 12?

### Questions
1. About the model analysis. Are there some explainable results showing that the model takes full advantage of the multivariable correlation while avoiding unnecessary noise?
2. As shown in Table 4. The performance promoted by CDAM is marginal (PatchTST on ETTh1). Is there any explanation for it? Besides, how does solely CDAM promote the performance, especially on datasets with more variables (such as Traffic).

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers utilizing information bottleneck as a regularizer for training Channel Mixing neural network models. The proposed idea aims to help learning better representations for multivariate time series, as the covariates may contain useful information about each other.
The developed approach augmented few state of the art models, and showed increased performance on several real-world datasets.

### Strengths
It is shown in experiments that the overfitting is indeed a problem and the proposed approach seems to be helping alleviating it. 

The ablation study where CDAM and TAM frameworks’ individual contributions are also shown to matter.

### Weaknesses
The objective function approximation seems to be a bit ad hoc, and the section on TAM component is not easy to follow. 

While TAM seems to be useful, not quite sure how downsampling and the additional approximations help. It would have been better to see a more detailed analysis of this step. 

Also it is not quite clear how the initial forecast was obtained.

### Questions
What is the purpose of Eq. (2)? Since the unconstrained form is used, I think that this discussion could start from Eq. (3).

The objective function approximation in Eq. (8) seems to be a bit ad hoc. It is not an upper bound on the entire objective function, but rather a combination of individual approximations. Discussion on this choice were not given in the paper. Is there some reason that makes approximation of the entire function prohibitively difficult? My main concern is that if these bounds are not tight the approximation could be way off.

Page 6, first paragraph: Is it CDAM or TAM?

Page 6: How is $\hat{Y}$ generated with a single step forecaster? Does that require an initial training of the neural network, or is this a different neural network that is trained before the actual network? Without an initial neural network, how can we obtain $\hat{Y}$ and train using Eq. (13)?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work aims to improve multivariate time series forecasting with transformers by focusing on the process of mixing channels and moving beyond single-step forecasting. They find that specifically modeling the relationships of channels with their proposed Cross-Variable Decorrelation Aware Feature Modeling can reduce the issues posed by indiscriminate channel-mixing or channel-independent methods.  Additionally, by modeling the temporal relations between subsequences in the forecast sequence with Temporal Correlation Aware Modeling, the authors create a framework for improving on existing work across standard benchmarks and various prediction lengths and model types.

### Strengths
Thanks to the authors for their submission: it contains useful research that shows good research practices while explaining an interesting and novel idea within multivariate time series forecasting. The results of this work will be informative to other researchers and are significant in improving our understanding of applying transformer methods to time series forecasting. Some specific strengths of this research:

- Novelty: InfoTime, which both captures cross-variable relationships and temporal dependencies is a novel approach to the multivariate time series forecasting problem that provides superior results to prior work. While I am uncertain of the novelty of the TAM part of the framework, I believe that combining these and the information theoretic approach of CDAM is new.
- Relevance: MTSF is a complex task, and the paper acknowledges and addresses two significant challenges: mixing channels in an efficient way and mitigating errors that accumulate over time. By providing specific solutions (CDAM and TAM), the paper contributes to improving the accuracy and effectiveness of MTSF models which are also used in many real-world tasks.
- The authors conduct comprehensive experimental analyses on real-world datasets to demonstrate the effectiveness of the InfoTime framework. These datasets are standard benchmarks in this area of research and the validation appears to be done through a very similar process to most other work making it easier to compare. The results consistently show that InfoTime outperforms existing Channel-mixing and channel-independence benchmarks, achieving superior accuracy and mitigating overfitting issues.
- Theory: The paper leverages concepts from information theory, such as Mutual Information and Information Bottleneck, to guide the development of CDAM and TAM. This theoretically grounded approach enhances the understanding and interpretability of the proposed framework.
- The paper was well written and provides a clear description of the proposed work and comparison with other methods.

### Weaknesses
1/ More comparison with statistical forecasting methods or non-transformer methods, either empirically or theoretically, could help introduce the benefits of InfoTime in a more thorough manner. While Zeng et. al (2022) and other methods are briefly discussed for their direct forecasting strategy; it could be helpful to compare InfoTime on the DLinear model to see how CAM varies from the DMS approach. Perhaps this is why RMLP was introduced, but it’s not clear what motivated the construction of RMLP and if differs at all from previous approaches which could be more suitable baseline models. 

2/ The implications of the lower bound for (Iv(Xi, Yi; Zi) are not clear, nor is the sampling strategy well motivated in the paper. Additional explanation of the implications could help to improve this.

3/ The experimentation work is not detailed. There are a number of factors that are unclear: the reported informer MSE and MAE on ETTh1, ETTm1, ETTm2 for example are quite a bit higher than in the original Informer, Non-stationary Transformer, and Crossformer papers (>1 for O=720 Informer by authors vs <0.3 in original paper). Perhaps more discussion of the training setup and how it might differ from past benchmarks and why it shows a much higher error could help separate out the actual performance gains given by InfoTime.

4/ Given the large tables of evaluations, it would be helpful to introduce some visual aides to help with understanding the comparisons, perhaps showing the % improvement over other methods for an aggregation of all benchmarks.

Nits:
sp. ‘varibales’ (pg. 5)

### Questions
1/ There is a lot of variation in results of benchmarks (ETTh1, ETTm1, ETTm2, Weather, Traffic, Electricity) across various papers. Have the authors considered running 5-fold cross validation on these benchmarks to better understand the error bounds around performance claims and then running tests to evaluate the statistical significance of the improvements?
2/ How does the time complexity of CDAM, TAM, and InfoTime compare with the original base models - particularly as the time series get longer?
3/ The paper introduces a trade-off parameter B in CDAM to control the balance between retaining important information and eliminating irrelevant information from the latent representation. How could this parameter be determined or optimized in practice? Are there any insights into choosing an appropriate value for B based on experimental results or theoretical considerations?
4/ The paper introduces a variational lower bound (Iv(Xi, Yi; Zi)) for the mutual information term in CDAM. Could you explain the practical implications of maximizing this lower bound during training? How does optimizing this lower bound improve the model's ability to capture cross-variable dependencies, and what trade-offs or challenges may arise in the optimization process?
5/ Additionally on the upper bound, the authors state that they choose to adopt the sampled vCLUB and minimize I_{vCLUB-S}(Xo; Zi). Could they expand on why this sampling strategy was chosen and what the actual derived upper bound might be?
6/ While the benchmarks used are standard for transformer-based evaluation, there are other baselines that could shed more light on the characteristics of the model and whether it may help resolve some of the difficulties with transformer approaches to forecasting. Could the authors shed any light on if they considered benchmarks like M3, M4 and what might be missing from the current evaluations?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
