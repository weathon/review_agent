# UniCA: Unified Covariate Adaptation for Time Series Foundation Model

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Time Series Foundation Models (TSFMs) have achieved remarkable success through large-scale pretraining. However, their design primarily targets real-valued series, limiting their ability to handle general forecasting tasks involving diverse and often \emph{heterogeneous covariates}—such as categorical variables and multimodal data (e.g., images, text)—which are typically task-specific and difficult to leverage during pretraining. To address this gap, we propose Unified Covariate Adaptation (UniCA), a framework to bridge TSFMs with general covariate-aware forecasting. UniCA first performs covariate homogenization to transform heterogeneous covariates into high-level homogeneous series representations and then fuses them via a unified attention-based fusion mechanism. UniCA is compatible and universal for adaptation with both homogeneous and heterogeneous covariates, incorporating extra covariate information while preserving the generalization ability of TSFMs. Extensive experiments on multiple unimodal and multimodal covariate-aware forecasting benchmarks demonstrate the superiority of UniCA, highlighting the promise of covariate-aware TSFM adaptation in real-world forecasting scenarios. Code: https://github.com/hanlu-nju/UniCA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed UniCA, which focuses on facilitating time series foundation models to be able to handle heterogeneous covariates, such as categorical variables and covariates from other modalities. This is achieved by projecting the covariates into the representation space. The representations are then fused with an attention mechanism.

### Strengths
**S1:** The motivation is clear. The research question highlighted by the paper is valid and addresses an important problem in time series forecasting for real-life scenarios, i.e., rather than relying on the target covariate itself, external covariates should be taken into account to better analyze the dynamics.

**S2:** The experiments are rather extensive. The way the authors try to present the results is straightforward to convey the information using visualized graphs.

### Weaknesses
**W1:** It makes sense to use a frozen pretrained model to transform the image and text into the representation space. However, when it comes to categorical data, the authors mentioned that they are converted using embedding layers, which have not been pretrained \& frozen. However, this embedding layer has no prior knowledge about the given categories. Rather, it serves as a lookup table of random vectors. This led to concern about whether the added categorical information is useful in training. See Q1 for follow-up.

**W2:** The exogenous covariates have different levels of information (e.g., the categorical is sparse while the image and text are dense). In this method, the inclusion of these external covariates has simply been stacked together. While this naive method is reasonable, it raises concerns that there might be cases where the dense information is down-weighted, limiting the model’s ability to fully leverage the high-dimensional semantic content. See Q2 for follow-up.

**W3:** Another concern would be that the covariates, after mapping with the embedding layer / model, do not contain temporal information that can be strictly aligned with the time series. From the modelling design perspective, the proposed architecture does not support the claim that the model captures the feature changes over time, as the linear layer is a mixing over time. 

**W4:** It would, in general, be beneficial if the authors could give more justification for the model design. Apart from the concern mentioned in W3. See Q3.

Minor: 

- Code in the anonymised link has expired.

- The notation system is confusing. The future-known and future-unknown covariates have been simplified without an explicit definition.

- The definition clarity can be improved. The term "Covariate" includes both the target and the exogenous variables (or as exogenous covariates, as the authors denoted in the paper). However, in some cases (e.g., lines 154-156), this paper simply uses covariates in place of exogenous covariates, which can lead to confusion.

- Some results can be hard to read due to an over small font, e.g., Fig 4 (c).

- The expression can be improved, e.g., a "-6.5\% improvement" can be "reduced the MAE by 6.5\%", otherwise it appears to be a performance degradation at first sight.

### Questions
**Q1:** Following W1, it would be useful to see the performance influence when the categorical information is mixed up / randomly initialised.

**Q2:** Following W2, it would be beneficial to also show how the performance would be influenced when the information is weighted differently w.r.t. their information level.

**Q3:** Do the two fusion methods, GRN (in pre-fusion) and attention (in post-fusion), work exchangeable or is there a special concern in this modelling choice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper formulates a novel problem setting, i.e., adapting pre-trained channel-independent time series foundation model to covariate-aware pratical tasks. The covariate may involve categorical variables,images or text modalities. The motivation is clear and significant.  Accordingly, a  Unified Covariate Adaptation (UniCA) framework is proposed, which bridges the gap between the pre-trained model and real practice. Extensive experiments are carried on to verify the effectiveness of the proposed approach.

### Strengths
1. The motivation is clear and sufficient. Adapting pre-trained channel-independent time series foundation model to covariate-aware pratical tasks makes sense.

2. This is the first work to formalize the problem of adapting Time Series Foundation Models (TSFMs) to general covariate-aware forecasting scenarios.

3. Technically sound and experiments are comprehensive.

### Weaknesses
1. When dealing with multi-modal covariate, the tokenization methods for different modality is critical, which need more discussion and explanation.

2. According to Fig.3 (a), the improvements for Chron os-Bolt and TimesFM seem to be marginal.

3. For modeling time series with discrete variables, there are existing works should be discussed:

[1]. General Mixed Time Series Analysis via Latent Continuity Recovery and Alignment. NeurIPS 2024.

### Questions
1. the code repo seems to be expired, could the author fix it?

2. The time series tasks addressed in this paper seem to be limited to forecastingonly. Can this framework be extended to other tasks, such as anomaly detection and classification?

3. Covariate-aware prediction is a very important scenario in practical time series applications. Moreover, actual time series data may also have issues such as data missingness. How does this framework handle these problems?

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
This paper proposes UniCA, a unified covariate adaptation framework that enables pretrained Time Series Foundation Models (TSFMs) to handle heterogeneous covariates such as categorical, image, and text data. By combining covariate homogenization and attention-based fusion, UniCA integrates diverse covariates without modifying pretrained TSFM parameters.

### Strengths
- The specially designed plug-in fusion modules preserve pretrained generalization while efficiently leveraging covariate information.
- The two-stage design (homogenization + fusion) is intuitive and broadly compatible with multiple TSFM architectures
- Strong empirical results demonstrating superior performance on 12 unimodal and multimodal datasets with minimal added cost.
- Comprehensive analysis and ablations confirming the framework’s universality, interpretability, and computational efficiency.

### Weaknesses
1. **Unclear embedding alignment across heterogeneous modalities.**  
  The paper claims that the *Covariate Homogenization* module transforms heterogeneous covariates (e.g., categorical, image, and text) into a unified latent space, yet no explicit *alignment loss* or *architectural constraint* ensures such consistency. Given the complexity of learned representations, the claim that all embeddings could be easily projected into the same latent space by only using one linear layer is not very convincing.

2. **Limited justification for architectural choices in CAP.**  
  It is unclear why the authors chose **Gated Residual Networks (GRN)** and **Gated Linear Units (GLU)** in the CAP instead of other modules. The paper should either explain the reasons why GRN and GLU are particularly suited for covariate fusion or provide ablation studies comparing them with simpler or alternative fusion mechanisms.

3. **Insufficient qualitative evidence for homogenization effectiveness.**  
  The visualization in Figure 5(b–c) shows only a single example, which may be cherry-picked. To convincingly demonstrate that the homogenized embeddings capture meaningful temporal structure, more examples or aggregate statistics should be included in the appendix.

4. **Lack of discussion on zero-shot forecasting capability.**  
  One of the defining advantages of TSFMs is their **zero-shot generalization**, yet the paper does not analyze whether UniCA preserves or degrades this property. Since UniCA introduces additional learned modules, a dedicated experiment or discussion on its impact on zero-shot forecasting is essential to establish its compatibility with the TSFM paradigm.

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose UniCA, Unified Covariate Adaptation, an adaptation method for Time Series Foundation Models (TSFMs) to support covariate-aware forecasting. This adaptation is achieved through addition of linear layers to adapt covariates into the FM embedding space and use of Conditional Attention Pooling (CAP) to fuse covariates signals into the FM univariate predictions.

Authors have conducted thorough evaluation using datasets containing time series covariates, as well as text and image covariates, and demonstrated their method outperforms prior work.

The main contribution of the work is the adaptation framework that supports any modality covariate-aware forecasting through training of the adapter.

### Strengths
1. Problem formulation and motivation are explained clearly.
2. Authors have conducted a thorough evaluation spanning various prior work, datasets and covariates.
3. Proposed method has minimal overhead during inference time.
4. Training one adapter per dataset can work on any forecast horizon. (No requirement of unique adapter for each forecast horizon)

### Weaknesses
1. Although TSFM parameters are frozen when training the adapter, training cost is still incurred to perform adaptation. This takes away the ability of TSFMs to forecast zero-shot. 
2. The adapter proposed in ChronosX is also based on linear layers. The contribution of this work therefore appears mostly incremental. The authors add additional parameters and attention modules before and after the backbone transformer FM. It is unclear whether the performance gain arises simply from these additional parameters.

### Questions
1. How does UniCA perform when provided with noisy covariates?
2. Can UniCA achieve 100% accuracy when provided with the oracle label (i.e. the target time series)?
3. Why does UniCA perform better with Chronos-bolt in some cases and with TimesFM in others?
4. Does your ChronosX implementation achieve the performance reported in the ChronosX Arango et al. paper?

### Soundness
3

### Presentation
3

### Contribution
2
