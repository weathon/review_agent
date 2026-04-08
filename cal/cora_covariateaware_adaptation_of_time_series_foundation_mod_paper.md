# CORA: COVARIATE-AWARE ADAPTATION OF TIME SERIES FOUNDATION MODELS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Time Series Foundation Models (TSFMs) have shown significant impact through
their model capacity, scalability, and zero-shot generalization. However, due to
the heterogeneity of inter-variate dependencies and the backbone scalability on
large-scale multivariate datasets, most TSFMs are typically pre-trained on univariate time series. This limitation renders them oblivious to crucial information from diverse covariates in real-world forecasting tasks. To further enhance
the performance of TSFMs, we propose a general **Co** variate-awa **R** e **A** daptation
( **CoRA** ) framework for TSFMs. It leverages pre-trained backbones of foundation models while effectively incorporating exogenous covariates from various
modalities, including time series, language, and images, to improve the quality of
predictions. Technically, CoRA maintains the equivalence of initialization and parameter consistency during adaptation. With preserved backbones of foundation
models as frozen feature extractors, the outcome embeddings from foundation
models are empirically demonstrated more informative than raw data. Further,
CoRA employs a novel Causality Embedding to automatically evaluate covariates
regarding their causal predictability with respect to the target variate. We incorporate these weighted embeddings with a zero-initialized condition-injection mechanism, avoiding catastrophic forgetting of pre-trained foundation models and gradually integrates exogenous information. Extensive experiments show that CoRA
of TSFMs surpasses state-of-the-art covariate-aware deep forecasters with full or
few-shot training samples, achieving 31 _._ 1% MSE reduction on covariate-aware
forecasting. Compared to other adaptation methods, CoRA exhibits strong com

|other modal<br>TSLib (Uni-Modal)<br>0.3<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx<br>0.2<br>0.186<br>0.177<br>MSE<br>0.137 0.137<br>0.1 0.094<br>0.0|Col2|Col3|
|---|---|---|
|<br>other modal<br>0.0<br>0.1<br>0.2<br>0.3<br>MSE<br>~~0.094~~<br>0.137 0.137<br>0.177<br>0.186<br>TSLib (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|
|<br>other modal<br>0.0<br>0.1<br>0.2<br>0.3<br>MSE<br>~~0.094~~<br>0.137 0.137<br>0.177<br>0.186<br>TSLib (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|~~0.094~~<br>0.137 0.137<br>0.177<br>0.186|~~0.094~~<br>0.137 0.137<br>0.177<br>0.186|
|<br>other modal<br>0.0<br>0.1<br>0.2<br>0.3<br>MSE<br>~~0.094~~<br>0.137 0.137<br>0.177<br>0.186<br>TSLib (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|||
|<br>other modal<br>0.0<br>0.1<br>0.2<br>0.3<br>MSE<br>~~0.094~~<br>0.137 0.137<br>0.177<br>0.186<br>TSLib (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|||


|EPF (Uni-Modal)<br>6<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>5<br>N-BEATSx<br>4<br>0.366<br>0.330 0.330<br>0.307<br>3 0.278<br>2<br>1<br>0|Col2|Col3|Col4|
|---|---|---|---|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|||CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|N-BEATSx|N-BEATSx|N-BEATSx|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|0.307<br>0.330 0.330<br>0.366|0.307<br>0.330 0.330<br>0.366|0.307<br>0.330 0.330<br>0.366|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx|0.278|0.278|0.278|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx||||
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx||||
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>0.278<br>0.307<br>0.330 0.330<br>0.366<br>EPF (Uni-Modal)<br>CoRA<br>TimeXer<br>PatchTST<br>DLinear<br>N-BEATSx||||


|RT-1 (Multi-Modal)|Col2|
|---|---|
|~~CoRA~~<br>TimesFM<br>Chronos-Bolt<br>PatchTST<br>DLinear|~~CoRA~~<br>TimesFM<br>Chronos-Bolt<br>PatchTST<br>DLinear|
|0.689|0.689|
|0.413<br>0.471<br>0.506<br>0.536|0.413<br>0.471<br>0.506<br>0.536|
|||
|||


|me-MMD (Multi-Moda|Col2|Col3|
|---|---|---|
|||CoRA<br>Moirai<br>N-BEATS<br>TabPFN-TS<br>|
|~~0.782 ~~0.787 0.793<br>PatchTST|~~0.782 ~~0.787 0.793<br>PatchTST|PatchTST|
|~~0.580~~<br>0.696<br>|~~0.580~~<br>0.696<br>|~~0.580~~<br>0.696<br>|
||||
||||
||||
||||


|TSLib (Multivariate)|Col2|Col3|
|---|---|---|
|||CoRA<br>TimeXer<br>PatchTST|
|DLinear<br>Crossformer|DLinear<br>Crossformer|DLinear<br>Crossformer|
|~~0.486~~|~~0.486~~|~~0.486~~|
||||
|0.300<br>0.350 0.362 0.366|0.300<br>0.350 0.362 0.366|0.300<br>0.350 0.362 0.366|
||||
||||
||||


Figure 1: CoRA performance on different covariate-aware benchmarks.


1


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.2


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


1 INTRODUCTION


Time series forecasting has gained increasing prominence in real-world applications, such as
weather forecasting (Hittawe et al., 2024), supply chain optimization (Panda & Mohanty, 2023) and
financial market assessment (Cheng et al., 2022). With the rapid development of large-scale timeseries datasets (Woo et al., 2023) and scalable architectures (Vaswani et al., 2017), recent research
has focused on developing Time Series Foundation Models (TSFMs) (Das et al., 2023b; Liu et al.,
2024c; Ansari et al., 2024; Liu et al., 2025), which exhibit impressive scalability and out-of-box
generalization performance across various applications.


Despite time series are typically multi-dimensional data, most TSFMs are pre-trained on univariate
time series (Das et al., 2023b; Liu et al., 2024d; Shi et al., 2024), primarily due to the considerable
heterogeneity in dimensionality and inter-variate relationships across datasets. In particular, the dependencies among variates in one dataset often fail to generalize to others. For example, transferring
relationships learned from meteorological variates to the financial domain may not be sensible. Besides, covariate-aware deep forecasters, which are trained in a channel-dependence approach (Qiu
et al., 2025), have not been well-demonstrated to be scalable and versatile. Meanwhile, an important paradigm of foundation models involves large-scale pre-training on general large-scale data and
adaptation to task-specific datasets. Therefore, these constraints necessitate the paradigm shift as
shown in Figure 2, which adapts TSFMs to covariate-aware forecasting scenarios while revitalizing
the pre-trained backbone of foundation models (Arango et al., 2025; Benechehab et al., 2025).


Different from adaptation methods for language models such as LoRA (Hu et al., 2021), covariateaware adaptation in time series forecasting faces fundamentally different challenges. The difficulty
lies in the multi-dimensionality and the heterogeneity of modalities in covariates. Simply incorporating exogenous information into the target variate is insufficient, because dependencies among
variates are often domain-specific, noncausal, and sometimes noisy. Therefore, adaptation of TSFM
requires not only the integration of covariate information but also evaluating the causality of different covariates. Guided by the principled criteria, we delve into Granger causality, a foundational
concept for identifying causal dependencies in time series forecasting (Granger, 1969), and develop
a date-dependent approach to ground covariate-aware adaptation with interpretable modular design.


While prior works (Arango et al., 2025; Benechehab et al., 2025; Han et al., 2025) attempt to incorporate time series covariates into TSFMs, they inject covariate-aware modules that alter the embeddings away from the pre-trained embedding space. Besides, previous adaptation methods introduce trainable modules without zero-initialization, implying that the initial outputs of the adapted
model are no longer equivalent to the pre-trained TSFMs. Empirically, adaptation without zeroinitialization will cause unstable training, catastrophic forgetting and sometimes even worse performance than just zero-shot evaluation (Hu et al., 2021; Peebles & Xie, 2023).


In this paper, we introduce **CoRA**, a general, effective, and interpretable framework to adapt TSFMs
on covariate-aware forecasting tasks, where covariates cover time series, language, images, and other
structured data. Concretely, CoRA treats pre-trained foundation models of different modalities as
frozen embedding extractors. With extracted embeddings from raw covariates, CoRA includes a covariate evaluation and routing module, termed Causality Embedding, which automatically produces
a causally-informed significance score during adaptation. These embeddings are then integrated


**(a) Covariate-Aware Training** **(b) Univariate Pre-training** **(c) Covariate-Aware Adaptation**


Channel Independence


Channel Dependence


Target Variates


Covariates (Time Series)


Covariates (Other Modalities)


Figure 2: Several paradigms of time series forecasting: (a) Covariate-aware deep models are supervisedly trained in a channel-dependent way. However, the backbone can be task-specific and
challenged to scale up. (b) TSFMs designed to address data heterogeneity are generally pre-trained
and predict on univariate time series. which makes them infeasible to utilize inter-variate dependencies explicitly. (c) CoRA leverages various foundation models, incorporates exogenous information
to predict the target variate, and rapidly adapts to specific tasks without altering pre-trained models.


2


through a zero-initialized condition-injection mechanism by learning scale and shift parameters.
CoRA achieves state-of-the-art performance while requiring fewer samples compared to supervised
models and previous adaptation methods. In-depth studies validate the generality and interpretability
of the proposed framework. Our main contributions are summarized as follows:


    - We emphasize that an important paradigm of covariate-aware forecasting on TSFMs, which
effectively revitalize pre-trained foundation models and address the unique challenges in
utilizing high-dimensional, multi-modal, and causally-dependent covariates.

    - We propose CoRA, a general and effective covariate-aware adaptation framework that
freezes pre-trained models and introduces a Causality Embedding for principled covariate selection, combined with a zero-initialized condition-injection mechanism.

    - Extensive experiments across diverse benchmarks demonstrate that CoRA achieves stateof-the-art performance, requires fewer training samples, and provides interpretable insights
into covariate causality, surpassing both supervised models and other adaptation methods.


2 RELATED WORK


2.1 TIME SERIES FOUNDATION MODELS


Recent research has explored pre-training Time Series Foundation Models (TSFMs) on large-scale
datasets, enabling strong zero-shot generalization to downstream tasks. TimesFM (Das et al., 2023b)
and Timer (Liu et al., 2024d) are the first to adopt a decoder-only Transformer architecture with the
next-token prediction objective. Chronos (Ansari et al., 2024) introduces a discretization approach
for time series and predicts next tokens using LLM backbone and language modeling. Sundial (Liu
et al., 2025) proposes TimeFlow, incorporating generative modeling to realize the flexibility of probabilistic forecasting. However, these models are limited to univariate pre-training, which restricts
their applicability to downstream tasks involving multi-dimensional or multi-modal covariates. One
exception is that Moirai (Woo et al., 2024) adopts multivariate pre-training by flattening variates and
appending variate-wise embeddings, but it has to subsample multivariate series with a fixed size for
training stability, leading to incomplete perception for high-dimensional time series inputs.


2.2 COVARIATE-AWARE DEEP FORECASTERS


In real-world time series forecasting, covariates play a crucial role in improving the predictability of
target variate. Classical approaches such as ARIMAX (Williams, 2001) and SARIMAX (Vagropoulos et al., 2016) model the correlations between covariates and the target variate by linear regression.
More recent deep learning methods, such as the Temporal Fusion Transformer (Lim et al., 2021),
emphasize variate selection as a key mechanism. Other approaches, including NBEATSx (Olivares
et al., 2023) and TiDE (Das et al., 2023a), argue that forecasting models can directly leverage future
covariate information when predicting target values. TimeXer (Wang et al., 2024) achieves competent performance by modeling the target variate at the patch level and the covariates at the series
level. Time-VLM (Zhong et al., 2025) leverages vision-language backbones to integrate temporal, visual, and textual information for multi-modal forecasting. However, supervised deep models
trained from scratch may yield suboptimal performance without substantial task-specific data.


2.3 ADAPATION METHODS OF FOUNDATION MODELS


Adaptation of foundation models such as LoRA (Hu et al., 2021; Dettmers et al., 2023) is typically
applied in language and vision models, where the upstream and downstream tasks share the same
1D-sequence structure. In contrast, adapting univariate pre-trained TSFMs to covariate-aware scenarios introduces dimensional changes in the input structure. Prior works such as ChronosX (Arango
et al., 2025), AdaPTS (Benechehab et al., 2025), and UniCA (Han et al., 2025) modify the TSFM
input structure by injecting covariates before the backbone, which inevitably alters the pre-trained
embedding space and may trigger catastrophic forgetting. In contrast, Gen-P-Tuning (Liu et al.,
2024b) learns covariate prompts at the front of the context, introducing a relatively smaller structural change. Moreover, adaptation of foundation models relies on zero-initialization (Goyal et al.,
2017) to ensure that the training start-point begins consistently with the pre-trained model. However,
such principled strategies have not been properly considered in existing TSFMs adaptation methods.


3


For the target variate, we use the TSFM backbone to extract its embeddings and take the embedding
at the last time step _T_ to capture the overall lookback information:

**E** [target] 1: _T_ = TSFM-Backbone( **x** 1: _T_ ) _,_ **E** [˜] [target] = **E** [target] _T_ _._ (4)

1Covariates may be future-unknown ( _τ_ = _T_ ), future-known ( _τ_ = _T_ + _H_ ), or static covariates ( _τ_ = 1).


4


Image Covariates


Figure 3: Overall architecture of CoRA. CoRA freezes the backbone of foundation models as embedding extractors for multi-modal covariates, which are then selected by a trainable Causality Embedding. This refined embedding is injected into the original TSFM head via a zero-initialized
module to generate the shifting and scaling factors for final predictions.


3 APPROACH


In covariate-aware forecasting, we consider one target variate **x** 1: _T_ = _{x_ 1 _, . . ., xT } ∈_ R _[T]_ observed
over _T_ time steps along with exogenous covariates **C** 1: _τ_ = _{_ **C** 1 _, . . .,_ **C** _τ_ _}_ [1] . The task is to train a
forecaster _fθ_ parameterized by _θ_ that can predict the target variate **x** _T_ +1: _T_ + _H_ = _{xT_ +1 _, . . ., xT_ + _H_ _}_
for the next _H_ time steps:
_fθ_ : ( **x** 1: _T,_ **C** 1: _τ_ ) _�→_ **x** ˆ _T_ +1: _T_ + _H_ _._ (1)


3.1 FOUNDATION MODELS AS FROZEN EMBEDDING EXTRACTOR


For real-world forecasting, exogenous covariates are very often multi-dimensional (e.g., multivariate
time series) and multi-modal. In contrast to previous methods that solely adapt the foundation model
of time series, we categorize exogenous covariates into three mainstream modalities. As illustrated
in Figure 3, we separate covariates as _N_ one-dimensional sequences, such as univariate time series,
text, or image snapshots, and extract per-step embeddings from corresponding frozen models:

**E** _[m]_ 1: _τ_ _[i]_ _i_ [= FM-Backbone(] **[C]** 1: _[m]_ _τ_ _[i]_ _i_ [)] _[,]_ _[i]_ [ = 1] _[, . . ., N,]_ _[m][i]_ _[∈{]_ [ts] _[,]_ [ txt] _[,]_ [ img] _[}][.]_ (2)


At each time step, the embeddings **E** [ts] _t_ _[∈]_ [R] _[N]_ [ts] _[×][D]_ [ts][,] **[ E]** _t_ [txt] _∈_ R _[N]_ [txt] _[×][D]_ [txt], and **E** [img] _t_ _∈_ R _[N]_ [img] _[×][D]_ [img]
capture the exogenous information of corresponding covariates by leveraging the embeddings generated before the last layer of the foundation models, where _D_ ts, _D_ txt, _D_ img denote the latent dimensions of the respective foundation models and _N_ ts, _N_ txt, _N_ img represent the number of covariates
categorized into each modality, with the total number of covariates _N_ = _N_ ts + _N_ txt + _N_ img.


For dynamic covariates that are recorded at each time step, CoRA regards one covariate as a whole
by aggregating the embeddings over all time steps. For typical TSFMs adopting the decoder-only
or encoder-decoder architecture, we employ the last-step embedding that corresponds to the latestknown values, which captures all previous context in one single-series covariate. For language and
vision foundation models that encode one snapshot, we utilize the averaged embeddings across all
snapshots of time steps (for simplicity, we omit the variate index _i_ ):


**E** ˜ [ts] = **E** [ts] _τ_ _[,]_ **[E]** [˜] [txt] [=] [1]

_τ_


_τ_


- **E** [txt] _t_ _[,]_ **[E]** [˜] [img] [=] [1]

_τ_

_t_ =1


_τ_


_τ_

- **E** [img] _t_ _._ (3)

_t_ =1


3.2 COVARIATE-AWARE ADAPTATION


**Granger** **Causality** Granger causality test (Granger, 1969) is a statistical hypothesis test used
to determine whether using a covariate **C** and **x** 1: _T_ to predict **x** _T_ +1: _T_ + _H_ yields a lower prediction error than using **x** 1: _T_ alone. If so, **C** is said to Granger causes **x** . Unlike real-world “whocauses-whom” causal relationships, Granger causality captures the predictive usefulness of _C_ for
forecasting **x** _T_ +1: _T_ + _H_, not whether _C_ is the true causal driver of **x** _T_ +1: _T_ + _H_ . A covariate can aid
prediction without directly causing **x** _T_ +1: _T_ + _H_ . For example, if a latent variable _y_ causes both _C_
and **x** _T_ +1: _T_ + _H_, _C_ may still improve the prediction of **x**, and thus be regarded as a Granger cause of
**x** _T_ +1: _T_ + _H_ . Granger causality also differs from simple correlations. For example, a sine and cosine
wave have zero correlation, yet the Granger causality test between them can be significant.


**Covariate Selection** In typical covariate-aware forecasting tasks, multiple covariates are involved,
and their significance of Granger causality with respect to the target variate may differ considerably.
Therefore, we introduce a trainable Causality Embedding **W** CE _∈_ R _[N]_, which learns to quantify the
causal influence of each covariate on **x** 1: _T_ . Empirically, we observe that the learned Causality Embedding exhibits highly consistent result with the statistical test of Granger causality in Section 4.2.
Concretely, we first align the embeddings of multi-modal covariates into a unified hidden space since
the latent dimensions of foundation models are not necessarily identical:
**E** ˆ _[m][i]_ = **E** ˜ _[m][i]_ **W** _[m][i]_ + **b** _[m][i]_ _,_ _i_ = 1 _, . . ., N,_ _mi_ _∈{_ ts _,_ txt _,_ img _},_

**E** ˆ = Concat      - **E** ˆ [ts] _,_ ˆ **E** [txt] _,_ ˆ **E** [img][�] _._ (5)


where **W** _[m][i]_ _∈_ R _[D][mi]_ _[×][D]_, **b** _[m][i]_ _∈_ R _[D]_ for _mi_ _∈{_ ts _,_ txt _,_ img _}_, and **E** [ˆ] _∈_ R _[N]_ _[×][D]_ . Afterwards, we use
Causality Embedding _W_ CE _∈_ R _[N]_ to evaluate and gate each covariate during the adaptation process,
yielding a unified embedding that aligns the latent space of TSFMs:

**H** = Softmax( **W** CE) _·_ **E** [ˆ] _._ (6)


**Covariate Injection** With obtained overall exogenous embeddings of all covariates, we adopt an
adaptive layer-normalization (adaLN) layer proposed by DiT (Peebles & Xie, 2023), which is widely
shown to outperform approaches such as concatenation and cross-attention on continuous-valued
modality. Specifically, **H** is mapped into _α_ _∈_ R _[H]_ and _β_, _γ_ _∈_ R _[D]_ via a lightweight MLP( _·_ ). The
outcomes are then applied via shift-and-scale operations to modulate the statistics before and after
the original head of TSFM, thereby injecting the covariate information into the adaptation process.
Finally, we adopt the identical loss function used in the pre-trained TSFM for training:

_γ, β, α_ = MLP             - **H**             - _,_


**Zero-Initialization** Similar to LoRA (Hu et al., 2021), we zero-initialize the parameters of
**W** _[m][i]_ _∈_ R _[D][mi]_ _[×][D]_, **b** _[m][i]_ _∈_ R _[D]_ for _mi_ _∈{_ ts _,_ txt _,_ img _}_ and the MLP. Therefore, the overall model
is identical to the pre-trained TSFM. This design ensures adaptation begins from the pre-trained
state, while progressively integrating additional information in a stable and incremental manner.


4 EXPERIMENTS


We conduct comprehensive experiments to evaluate the effectiveness of CoRA, covering uni-modal
and multi-modal covariate-aware forecasting, few-shot forecasting, and extensions to multivariate
forecasting. The overall performance is provided in Figure 1. We further provide in-depth analysis,
including generality across different TSFMs, ablation studies, and model interpretability.


4.1 MAIN RESULTS


In this section, we conduct extensive experiments to evaluate the performance of CoRA, compared
with existing adaptation methods and advanced supervised deep forecasters. For fair comparison,
we adopt Sundial (Liu et al., 2025) as the backbone model for all adaptation approaches. Moreover,
we ensure none of the test sets overlap with Sundial’s training data to avoid potential data leakage.


5


**x** ˆ _T_ +1: _T_ + _H_ = (1 + _α_ ) TSFM-Head - _γ_ + (1 + _β_ ) **E** [˜] [target][�] _._


loss=TSFM-Loss(ˆ **x** _T_ +1: _T_ + _H_ _,_ **x** _T_ +1: _T_ + _H_ )


(7)


Table 1: Averaged results of the long-term covariate-aware forecasting. For all baselines, the lookback length _L_ is fixed at 2880. The reported performance is averaged over prediction horizons _S_ =
_{_ 96, 192, 336, 720 _}_ and full results are provided in Table 8. Dash (-) denotes out of memory.


**CoRA** AdaPTS ChronosX UniCA TimeXer iTransformer PatchTST NBEATSx Crossformer DLinear
Models
**(Ours)** (2025) (2025) (2025) (2024) (2023) (2022) (2023) (2023) (2023)

|Metric|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|
|---|---|---|---|---|---|---|---|---|---|---|
|ETTh1|**0.068 0.203**|0.076 0.211|0.085 0.227|0.085 0.222|0.089 0.240|0.160 0.317|0.096 0.249|0.181 0.351|0.386 0.501|0.263 0.408|
|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|ETTh2 **0.141 0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454|
|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|ETTm1 **0.043 0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184|
|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|ETTm2 **0.100 0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266|
|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|Weather **0.001 0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076|
|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|ECL<br>**0.194 0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376|
|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|Traffc **0.112 0.186**<br>-<br>-<br>-<br>-<br>0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317|


Table 2: Full results of the short-term covariate-aware forecasting. Following the standard protocol
of EPF dataset, with input-output lengths of 168-24. Avg means the average results from all five
datasets. Results of end-to-end models are officially reported by TimeXer (Wang et al., 2024).


**CoRA** AdaPTS UniCA ChronosX TimeXer iTransformer PatchTST NBEATSx Crossformer DLinear
Models
**(Ours)** (2025) (2025) (2025) (2024) (2023) (2022) (2023) (2023) (2023)

|Metric|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|
|---|---|---|---|---|---|---|---|---|---|---|
|NP|**0.222 0.246**|0.231 0.259|0.265 0.289|0.254 0.278|0.236 0.268|0.265 0.300|0.267 0.284|0.272 0.301|0.240 0.285|0.309 0.321|
|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|PJM<br>**0.073 0.165** 0.080 0.173 0.090 0.187 0.089 0.189 0.093 0.192 0.097 0.197 0.106 0.209 0.097 0.189 0.101 0.199 0.108 0.215|
|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|BE<br>**0.339 0.236** 0.355 0.261 0.368 0.273 0.371 0.274 0.379 0.243 0.394 0.270 0.400 0.262 0.389 0.265 0.420 0.290 0.463 0.313|
|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|FR<br>**0.357 0.206** 0.363 0.218 0.365 0.218 0.361 0.217 0.385 0.208 0.439 0.233 0.411 0.220 0.393 0.211 0.434 0.208 0.429 0.260|
|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|DE<br>**0.401 0.388** 0.455 0.424 0.553 0.466 0.453 0.426 0.440 0.415 0.479 0.443 0.461 0.432 0.499 0.447 0.574 0.430 0.520 0.463|
|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|AVG<br>**0.278 0.248** 0.297 0.267 0.328 0.287 0.306 0.277 0.307 0.265 0.335 0.289 0.330 0.282 0.330 0.283 0.354 0.284 0.366 0.314|


4.1.1 UNI-MODAL COVARIATE-AWARE FORECASTING


**Setups** In the uni-modal setting, all covariates are time series. We conduct both long-term and
short-term uni-modal covariate-aware forecasting experiments. In the long-term setting, we use
seven real-world datasets, including ECL, ETT (4 subsets), Traffic, and Weather, employed in Autoformer (Wu et al., 2021), where the final dimension serves as the target variate and the remaining
dimensions as covariates. In the short-term setting, we adopt the electricity price forecasting (EPF)
task (Lago et al., 2021), with electricity price as the target variate and two correlated covariates.


**Results** As shown in Table 1 and Table 2, CoRA delivers state-of-the-art performance across both
long- and short-term forecasting. Specifically, in long-term forecasting, CoRA outperforms the
strongest supervised model TimeXer (Wang et al., 2024), by 31.1% in MSE and 19.8% in MAE,
stressing the advantage of building on pre-trained TSFMs rather than training task-specific models
from scratch. Compared to other adaptation methods, using the same model Sundial (Liu et al.,
2025), CoRA reduces MSE by 18.7% compared to the second best adaptation method UniCA (Han
et al., 2025), highlighting the importance of maintaining parameter consistency and equivalent initialization during adaptation. In the EPF task, CoRA reduces MSE by 9.4% compared to TimeXer
and by 6.4% compared to AdaPTS (Benechehab et al., 2025), further solidifying its position as a
superior and generalized approach for uni-modal covariate-aware forecasting.


4.1.2 MULTI-MODAL COVARIATE-AWARE FORECASTING


**Setups** We evaluate CoRA on tasks involving multi-modal covariates, specifically images and
text. For image-based covariates, we construct a subset from the RT-1 (Brohan et al., 2022) dataset,
which contains a target time series with image covariates at each timestamp. For text-based covariates, we choose the Time-MMD (Liu et al., 2024a) dataset, which includes a target time series


6


Deep Models Pre-trained Model Zero-Shot Pre-trained Model Adaptation


Figure 4: Multi-modal covariate-aware forecasting on a subset of RT-1 (Brohan et al., 2022) with a
time series target variate and an image covariate. Input length is set to 32 and prediction length is 4.


Table 3: Multi-modal covariate-aware forecasting on Time-MMD ( **?** ) with textual covariates. Baseline results are reported by UniCA (Han et al., 2025), with full results in Table 9.

|Models|CoRA UniCA Sundial Moirai TabPFN-TS PatchTST TTM TiDE N-BEATS TFT DeepAR<br>(Ours) (2025) (2025) (2024) (2025) (2022) (2024) (2023a) (2023) (2021) (2020)|
|---|---|
|Average<br>MSE<br>MAE<br>CRPS|**0.641**<br>0.661<br>0.662<br>0.751<br>0.795<br>0.933<br>0.820<br>0.927<br>0.882<br>0.947<br>1.361<br>**0.580**<br>0.591<br>0.591<br>0.696<br>0.787<br>0.793<br>0.685<br>0.869<br>0.782<br>0.992<br>1.605<br>**0.690**<br>0.716<br>0.716<br>0.821<br>0.837<br>1.009<br>0.866<br>0.976<br>0.884<br>0.958<br>1.219<br>**0.653**<br>0.677<br>0.678<br>0.735<br>0.762<br>0.996<br>0.909<br>0.937<br>0.980<br>0.891<br>1.260|


with a corresponding text covariate. Moreover, CoRA adopts ViT [2] (Wu et al., 2020) and Qwen3Embedding [3] (Zhang et al., 2025) as backbone to extract features from image and text respectively.


**Results** As shown in Figure 4 and Table 3, CoRA achieves state-of-the-art performance across
all metrics. On the RT-1 ( **?** ) dataset, CoRA outperforms the best end-to-end supervised model and
TSFM zero-shot by 12.7% in MSE and 8.8% in CRPS. While on the Time-MMD benchmark ( **?** ), the
improvements are 1.9% in MSE and 3.7% in CRPS. These results demonstrate that properly modeling auxiliary modalities provides substantial benefits for forecasting. Compared with UniCA (Han
et al., 2025), which does not maintain backbone consistency or use proper zero-initialization, CoRA
consistently achieves superior performance on both benchmarks.


4.1.3 FEW-SHOT FORECASTING


**Setups** In real-world applications, the available training data is often highly limited, making
few-shot forecasting a critical challenge for robust deployment. We evaluate CoRA on the wellestablished electricity price forecasting (EPF) task (Lago et al., 2021), comparing it with alternative
adaptation methods and end-to-end models across a range of data scarcity levels.


Figure 5: Few-shot forecasting on the EPF dataset, comparing CoRA with TimeXer (Wang et al.,
2024) and ChronosX (Arango et al., 2025) across different levels of data availability.


**Results** As shown in Figure 5, CoRA consistently outperforms TimeXer (Wang et al., 2024) and
ChronosX (Arango et al., 2025) under different data availability levels. When the number of samples


[2https://huggingface.co/google/vit-base-patch16-224-in21k.](https://huggingface.co/google/vit-base-patch16-224-in21k)
[3https://huggingface.co/Qwen/Qwen3-Embedding-0.6B.](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)


7


is particularly small (1% to 25%), the end-to-end model TimeXer performs significantly worse than
adaptation methods based on pre-trained TSFMs, highlighting that pre-trained models can adapt to
downstream tasks more quickly and effectively with limited data. Even with sufficient data, TimeXer
still underperforms compared with adaptation methods, due to its relatively smaller model capacity.
Moreover, thanks to principled designs that preserve the pre-trained backbone and employ proper
zero-initialization, CoRA consistently outperforms ChronosX.


4.1.4 MULTIVARIATE TIME SERIES FORECASTING


**Setups** CoRA naturally extends to the multivariate time series forecasting scenarios via the
channel-independence mechanism, enabling joint prediction of multiple target variates. We evaluate this on seven real-world datasets introduced in Autoformer (Wu et al., 2021).


**Results** As shown in Table 4, CoRA outperforms all other supervised forecasters, achieving average MSE and MAE reductions of 14.5% and 12.2% compared to TimeXer (Wang et al., 2024).
CoRA’s superior performance stems from its use of pre-trained TSFMs that have already internalized universal temporal patterns from large-scale datasets. This enables CoRA to more accurately
capture inter-variate dependencies and generalize effectively across diverse datasets.


Table 4: Averaged results of the multivariate forecasting task on well-acknowledged benchmarks.
For all baselines, the look-back length _L_ is fixed at 2880. The reported performance is averaged
over prediction horizons _S_ = _{_ 96, 192, 336, 720 _}_ and full results are provided in Table 10.

|CoRA Timer-XL TimeXer iTransformer PatchTST Crossformer TiDE DLinear SCINet Autoformer<br>Models<br>(Ours) (2024d) (2024) (2023) (2022) (2023) (2023a) (2023) (2022) (2021)<br>Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|ETTh1|**0.404 0.422**|0.548 0.547|0.492 0.488|0.508 0.515|0.516 0.504|0.643 0.594|0.656 0.587|0.519 0.512|0.780 0.660|0.812 0.661|
|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|ETTh2 **0.331 0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707|
|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|ETTm1 **0.337 0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682|
|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|ETTm2 **0.256 0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495|
|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|Weather **0.230 0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487|
|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|ECL<br>**0.155 0.250 0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390|
|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|Traffc **0.384 0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464|


4.2 MODEL ANALYSIS


In this section, we perform thorough experiments to analyze several properties of CoRA, including
its generalization to other TSFMs such as TimesFM (Das et al., 2023b), Chronos-bolt (Ansari et al.,
2024), and FlowState (Graf et al., 2025), ablation studies on the method’s key components, and the
interpretability of learned Causality Embedding.


**Generality** Figure 6 shows that CoRA further boosts the performance of various TSFMs on top
of their zero-shot results. Average MSE reductions are 14.2% on Sundial (Liu et al., 2025), 3.3%
on TimesFM (Das et al., 2024), 4.9% on Chronos-Bolt (Ansari et al., 2024), and 3.3% on FlowState (Graf et al., 2025). These results demonstrate that CoRA offers an effective and flexible adaptation strategy, seamlessly integrating with diverse backbone architectures.


**Ablation Study** We provide a thorough ablation study to examine our proposed CoRA in Table 5.
Our results show that each component is crucial for CoRA’s performance by addressing specific
challenges in covariate-aware time series forecasting. Without the covariates’ information, forecasting performance degrades, underscoring the necessity of incorporating external signals to enhance
the predictability of the target. Without the adaLN module, we find that simply adding the condition
to the TSFM head input is insufficient. Instead, our condition-injection mechanism is highly effective by influencing the statistics of the TSFM head to fuse information. Similarly, when we removed
the Causality Embedding, replacing it with mean aggregation, the model’s performance dropped.
This demonstrates the importance of our selection and routing mechanism, which automatically
assigns appropriate weights to different covariates based on their inherent causality. Finally, we


8


|Figure<br>observ<br>This c<br>pre-tra<br>Table<br>withou<br>to the<br>aggreg|Col2|Col3|Col4|ance gains of CoRA across diverse TSFM<br>acing zero-initialization with Xavier initi<br>zero-initialization is vital for preserving<br>nsuring a stable adaptation process.<br>study of CoRA. (1) w/o covariate denotes<br>ariates. (2) w/o adaLN replaces the adaLN<br>e TSFM head. (3) w/o selection replace<br>/o zero-init replaces zero-initialization wi|Col6|Col7|Col8|Col9|Col10|ts are provided in Table 1<br>lted in worse performan<br>knowledge learned duri<br>Fine-Tuning (SFT), train<br>irectly adding the conditi<br>ity Embedding with me<br>tialization.|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Figure<br>observ<br>This c<br>pre-tra<br>Table<br>withou<br>to the <br>aggreg|||||||||||||
|Figure<br>observ<br>This c<br>pre-tra<br>Table<br>withou<br>to the <br>aggreg||6<br>ed<br> on<br>in<br> 5: <br>t u<br> in<br>at|: Perform<br> that repl<br> frms that<br>ing and e<br> Ablation<br> sing cov<br>put of th<br>ion. (4)_ w_|: Perform<br> that repl<br> frms that<br>ing and e<br> Ablation<br> sing cov<br>put of th<br>ion. (4)_ w_|: Perform<br> that repl<br> frms that<br>ing and e<br> Ablation<br> sing cov<br>put of th<br>ion. (4)_ w_|: Perform<br> that repl<br> frms that<br>ing and e<br> Ablation<br> sing cov<br>put of th<br>ion. (4)_ w_|: Perform<br> that repl<br> frms that<br>ing and e<br> Ablation<br> sing cov<br>put of th<br>ion. (4)_ w_|s. Full re<br>   alization<br>    the valu<br>  Supervi<br>   module b<br>s the Ca<br>  th Xavie|sul<br>    resu<br>     able<br>  sed<br>   y d<br>usal<br>   r ini|sul<br>    resu<br>     able<br>  sed<br>   y d<br>usal<br>   r ini|sul<br>    resu<br>     able<br>  sed<br>   y d<br>usal<br>   r ini|sul<br>    resu<br>     able<br>  sed<br>   y d<br>usal<br>   r ini|
|Da<br>M|Da<br>M|tas<br>od|ets<br>els<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|FR|E|DE<br>Avg<br>MSE<br>MAE<br>MSE<br>MA|DE<br>Avg<br>MSE<br>MAE<br>MSE<br>MA|DE<br>Avg<br>MSE<br>MAE<br>MSE<br>MA|
|Da<br>M|Da<br>M|tas<br>od|ets<br>els<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|NP<br>PJM<br>BE<br>SE<br>MAE<br>MSE<br>MAE<br>MSE<br>MAE<br>M|SE<br>MA|SE<br>MA|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|
|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|**CoRA**<br>**0.222**<br>**0.246**<br>**0.073**<br>**0.165**<br>**0.339**<br>**0.236**<br>**0.**|<br>**0.20**|**6**<br>|**0.401**<br>**0.388**<br>**0.278**<br>**0.24**|**0.401**<br>**0.388**<br>**0.278**<br>**0.24**|**0.401**<br>**0.388**<br>**0.278**<br>**0.24**|
|w/o covariate<br>0.2|w/o covariate<br>0.2|w/o covariate<br>0.2|w/o covariate<br>0.2|31<br>0.256|0.078<br>0.172<br>0.|0.078<br>0.172<br>0.|352<br>0.262<br>0.|360<br>0.21|4<br>|0.458<br>0.426|0.296<br>0.26|0.296<br>0.26|
|w/o adaLN<br>0.2|w/o adaLN<br>0.2|w/o adaLN<br>0.2|w/o adaLN<br>0.2|60<br>0.288|0.085<br>0.180<br>0.|0.085<br>0.180<br>0.|351<br>0.238<br>0.|368<br>0.21|0<br>|0.506<br>0.451|0.314<br>0.27|0.314<br>0.27|
|w/o selection<br>0.2|w/o selection<br>0.2|w/o selection<br>0.2|w/o selection<br>0.2|73<br>0.266|0.080<br>0.177<br>0.|0.080<br>0.177<br>0.|356<br>0.262<br>0.|360<br>0.21|5<br>|0.472<br>0.423|0.301<br>0.26|0.301<br>0.26|
|w/o zero-init<br>0.2|w/o zero-init<br>0.2|w/o zero-init<br>0.2|w/o zero-init<br>0.2|34<br>0.262|0.078<br>0.173<br>0.|0.078<br>0.173<br>0.|350<br>0.257<br>0.|360<br>0.20|8<br>|0.430<br>0.415|0.290<br>0.26|0.290<br>0.26|
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||


Figure 7: Correlation between traditional statistic Granger-Geweke Causality (Dhamala et al., 2018)
and the Causality Embedding learned in CoRA on ETTh1 Dataset.


**Interpretability** To study the interpretability of CoRA, we compare the learned Causality Embedding with the traditional Granger-Geweke Causality (Dhamala et al., 2018). We select 1000
windows from the ETTh1 dataset and compute the Granger-Geweke Causality for each window
(detailed description in the Algorithm 2) as well as the Causality Embedding learned by CoRA. Figure 7 demonstrates a strong correlation between the Granger–Geweke Causality and the Causality
Embedding. Furthermore, we plot a histogram of the Pearson correlation coefficient (Pearson, 1895)
across the 1000 windows, which clearly demonstrates their consistency.


5 CONCLUSION


In this paper, we introduce CoRA, a general, flexible, and interpretable framework for adapting pretrained foundation models to covariate-aware forecasting tasks. An important paradigm of foundation models involves large-scale pre-training on general datasets followed by adaptation to task

9


specific datasets. CoRA leverages this paradigm by using the powerful backbones of diverse foundation models as frozen embedding extractors. It then employs a Causality Embedding to weight and
select covariates based on their causal relationship to the target variate, and a zero-initialized adaLN
module for stable and progressive fusion of this information. Our extensive experiments consistently show that CoRA outperforms both advanced supervised models and other adaptation methods
while requiring fewer training samples, bridging the gap between powerful pre-trained models and
the complex multi-modal and multivariate challenges of real-world scenarios.


REFERENCES


Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen,
Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al.
Chronos: Learning the language of time series. _arXiv preprint arXiv:2403.07815_, 2024.


Sebastian Pineda Arango, Pedro Mercado, Shubham Kapoor, Abdul Fatir Ansari, Lorenzo Stella,
Huibin Shen, Hugo Senetaire, Caner Turkmen, Oleksandr Shchur, Danielle C Maddix, et al.
Chronosx: Adapting pretrained time series models with exogenous variables. _arXiv_ _preprint_
_arXiv:2503.12107_, 2025.


Abdelhakim Benechehab, Vasilii Feofanov, Giuseppe Paolo, Albert Thomas, Maurizio Filippone,
and Bal´azs K´egl. Adapts: Adapting univariate foundation models to probabilistic multivariate
time series forecasting. _arXiv preprint arXiv:2502.10235_, 2025.


Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn,
Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics
transformer for real-world control at scale. _arXiv preprint arXiv:2212.06817_, 2022.


Dawei Cheng, Fangzhou Yang, Sheng Xiang, and Jin Liu. Financial time series forecasting with
multi-modality graph neural network. _Pattern Recognition_, 121:108218, 2022.


Abhimanyu Das, Weihao Kong, Andrew Leach, Rajat Sen, and Rose Yu. Long-term forecasting
with tide: Time-series dense encoder. _arXiv preprint arXiv:2304.08424_, 2023a.


Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for
time-series forecasting. _arXiv preprint arXiv:2310.10688_, 2023b.


Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for
time-series forecasting. In _Forty-first International Conference on Machine Learning_, 2024.


Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In A. Oh, T. Naumann, A. Globerson,
K. Saenko, M. Hardt, and S. Levine (eds.), _Advances_ _in_ _Neural_ _Information_ _Pro-_
_cessing_ _Systems_, volume 36, pp. 10088–10115. Curran Associates, Inc., 2023. URL
[https://proceedings.neurips.cc/paper_files/paper/2023/file/](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
[1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)


Mukesh Dhamala, Hualou Liang, Steven L Bressler, and Mingzhou Ding. Granger-geweke causality: Estimation and interpretation. _NeuroImage_, 175:460–463, 2018.


Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Sumanta Mukherjee, Nam Nguyen, Wesley M
Gifford, Chandra Reddy, and Jayant Kalagnanam. Tiny time mixers (ttms): Fast pre-trained
models for enhanced zero/few-shot forecasting of multivariate time series. _Advances_ _in_ _Neural_
_Information Processing Systems_, 37:74147–74181, 2024.


Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet
in 1 hour. _arXiv preprint arXiv:1706.02677_, 2017.


Lars Graf, Thomas Ortner, StanisL [´] WoLs¸niak, [´] Angeliki Pantazi, et al. Flowstate: Sampling rate
invariant time series forecasting. _arXiv preprint arXiv:2508.05287_, 2025.


Clive WJ Granger. Investigating causal relations by econometric models and cross-spectral methods.
_Econometrica:_ _journal of the Econometric Society_, pp. 424–438, 1969.


10


Lu Han, Yu Liu, Qiwen Deng, Jian Jiang, Yinbo Sun, Zhe Yu, Binfeng Wang, Xingyu Lu, Lintao
Ma, Han-Jia Ye, et al. Unica: Adapting time series foundation model to general covariate-aware
forecasting. _arXiv preprint arXiv:2506.22039_, 2025.


Mohamad Mazen Hittawe, Fouzi Harrou, Mohammed Amine Togou, Ying Sun, and Omar Knio.
Time-series weather prediction in the red sea using ensemble transformers. _Applied Soft Comput-_
_ing_, 164:111926, 2024.


Shi Bin Hoo, Samuel M¨uller, David Salinas, and Frank Hutter. From tables to time: How tabpfn-v2
outperforms specialized time series forecasting models. _arXiv preprint arXiv:2501.02945_, 2025.


Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv_ _preprint_
_arXiv:2106.09685_, 2021.


Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, et al. Time-llm: Time series forecasting by reprogramming
large language models. _arXiv preprint arXiv:2310.01728_, 2023.


Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, and Rafał Weron. Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark.
_Applied Energy_, 293:116983, 2021.


Bryan Lim, Sercan O [¨] Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for
interpretable multi-horizon time series forecasting. _International Journal of Forecasting_, 37(4):
1748–1764, 2021.


Haoxin Liu, Shangqing Xu, Zhiyuan Zhao, Lingkai Kong, Harshavardhan Prabhakar Kamarthi,
Aditya Sasanur, Megha Sharma, Jiaming Cui, Qingsong Wen, Chao Zhang, et al. Time-mmd:
Multi-domain multimodal dataset for time series analysis. _Advances in Neural Information Pro-_
_cessing Systems_, 37:77888–77933, 2024a.


Mingzhu Liu, Angela H Chen, and George H Chen. Generalized prompt tuning: Adapting frozen
univariate time series foundation models for multivariate healthcare time series. _arXiv_ _preprint_
_arXiv:2411.12824_, 2024b.


Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet:
Time series modeling and forecasting with sample convolution and interaction. _Advances_ _in_
_Neural Information Processing Systems_, 35:5816–5828, 2022.


Xu Liu, Juncheng Liu, Gerald Woo, Taha Aksu, Yuxuan Liang, Roger Zimmermann, Chenghao
Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Moirai-moe: Empowering time series
foundation models with sparse mixture of experts. _arXiv preprint arXiv:2410.10469_, 2024c.


Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long.
itransformer: Inverted transformers are effective for time series forecasting. _arXiv_ _preprint_
_arXiv:2310.06625_, 2023.


Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer-xl: Longcontext transformers for unified time series forecasting. _arXiv preprint arXiv:2410.04803_, 2024d.


Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, and
Mingsheng Long. Sundial: A family of highly capable time series foundation models. _arXiv_
_preprint arXiv:2502.00816_, 2025.


Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64
words: Long-term forecasting with transformers. _arXiv preprint arXiv:2211.14730_, 2022.


Kin G Olivares, Cristian Challu, Grzegorz Marcjasz, Rafał Weron, and Artur Dubrawski. Neural
basis expansion analysis with exogenous variables: Forecasting electricity prices with nbeatsx.
_International Journal of Forecasting_, 39(2):884–900, 2023.


11


Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis
expansion analysis for interpretable time series forecasting. _arXiv_ _preprint_ _arXiv:1905.10437_,
2019.


Sandeep Kumar Panda and Sachi Nandan Mohanty. Time series forecasting and modeling of food
demand supply chain based on regressors analysis. _Ieee Access_, 11:42679–42700, 2023.


Karl Pearson. Vii. note on regression and inheritance in the case of two parents. _proceedings of the_
_royal society of London_, 58(347-352):240–242, 1895.


William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of_
_the IEEE/CVF international conference on computer vision_, pp. 4195–4205, 2023.


Xiangfei Qiu, Hanyin Cheng, Xingjian Wu, Jilin Hu, Chenjuan Guo, and Bin Yang. A comprehensive survey of deep learning for multivariate time series forecasting: A channel strategy perspective. _arXiv preprint arXiv:2502.10721_, 2025.


David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. _International journal of forecasting_, 36(3):1181–
1191, 2020.


Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Timemoe: Billion-scale time series foundation models with mixture of experts. _arXiv_ _preprint_
_arXiv:2409.16040_, 2024.


Stylianos I Vagropoulos, GI Chouliaras, Evaggelos G Kardakos, Christos K Simoglou, and Anastasios G Bakirtzis. Comparison of sarimax, sarima, modified sarima and ann-based models for
short-term pv generation forecasting. In _2016 IEEE international energy conference (ENERGY-_
_CON)_, pp. 1–6. IEEE, 2016.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances_ _in_ _neural_ _informa-_
_tion processing systems_, 30, 2017.


Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Yong Liu, Yunzhong Qiu, Haoran Zhang, Jianmin Wang,
and Mingsheng Long. Timexer: Empowering transformers for time series forecasting with exogenous variables. _arXiv preprint arXiv:2402.19072_, 2024.


Billy M Williams. Multivariate vehicular traffic flow prediction: Evaluation of arimax modeling.
_Transportation Research Record_, 1776(1):194–200, 2001.


Gerald Woo, Chenghao Liu, Akshat Kumar, and Doyen Sahoo. Pushing the limits of pre-training
for time series forecasting in the cloudops domain. _arXiv preprint arXiv:2310.05063_, 2023.


Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. _arXiv_ _preprint_
_arXiv:2402.02592_, 2024.


Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Zhicheng Yan, Masayoshi
Tomizuka, Joseph Gonzalez, Kurt Keutzer, and Peter Vajda. Visual transformers: Token-based
image representation and processing for computer vision, 2020.


Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, 34:22419–22430, 2021.


Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series
forecasting? In _Proceedings_ _of_ _the_ _AAAI_ _conference_ _on_ _artificial_ _intelligence_, volume 37, pp.
11121–11128, 2023.


Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. _arXiv preprint arXiv:2506.05176_,
2025.


12


Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency
for multivariate time series forecasting. In _The_ _Eleventh_ _International_ _Conference_ _on_ _Learning_
_Representations_, 2022.


Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency
for multivariate time series forecasting. In _The_ _eleventh_ _international_ _conference_ _on_ _learning_
_representations_, 2023.


Siru Zhong, Weilin Ruan, Ming Jin, Huan Li, Qingsong Wen, and Yuxuan Liang. Time-vlm: Exploring multimodal vision-language models for augmented time series forecasting. _arXiv preprint_
_arXiv:2502.04395_, 2025.


Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
Informer: Beyond efficient transformer for long sequence time-series forecasting. In _Proceedings_
_of the AAAI conference on artificial intelligence_, volume 35, pp. 11106–11115, 2021.


13


A EXPERIMENTAL DETAILS


A.1 DATASETS


To comprehensively evaluate the performance of CoRA, we conduct extensive experiments on several well-established benchmarks. The evaluation covers uni-modal, multi-modal covariate-aware
forecasting and multivariate forecasting tasks. The datasets we used are described below:


For uni-modal, long-term, covariate-aware forecasting tasks, we include the following benchmark
datasets: ETT (Electricity Transforming Temperature) (Zhou et al., 2021) contains seven power
transformer load factors from July 2016 to July 2018. According to sampling frequency and location, the dataset is partitioned into four subsets: ETTh1 and ETTh2 contain hourly measurements,
whereas ETTm1 and ETTm2 provide observations at 15-minute intervals. Weather (Wu et al., 2021)
comprises 21 meteorological variates collected at 10-minute intervals throughout 2020 from the Max
Planck Institute for Biogeochemistry. ECL (Electricity Consuming Load) (Wu et al., 2021) records
hourly electricity consumption of 321 residential and commercial clients, offering diverse patterns
of consumption behavior. Traffic (Wu et al., 2021) consists of hourly road occupancy data from 862
sensors installed on highways in the San Francisco Bay Area, covering the period January 2015 to
December 2016. Further statistics are reported in Table 6.


For uni-modal short-term covariate-aware forecasting task, we include the following benchmark
datasets: EPF (Electricity Price Forecasting) (Lago et al., 2021) contains 6 years of hourly dayahead electricity prices, complemented by two exogenous forecast series (load and renewable generation). The dataset spans five major European electricity markets, facilitating robust cross-market
performance analysis under diverse price dynamics and market conditions. (1) NP (Nord Pool)
covers the Nord Pool electricity market, containing hourly electricity prices together with grid load
and wind power forecasts from 2013-01-01 to 2018-12-24. (2) PJM corresponds to the Pennsylvania–New Jersey–Maryland market, including the zonal electricity price in the Commonwealth
Edison (COMED) area, system load, and COMED load forecasts from 2013-01-01 to 2018-12-24.
(3) BE denotes Belgium’s electricity market, recording hourly electricity prices, load forecasts in
Belgium, and generation forecasts in France from 2011-01-09 to 2016-12-31. (4) FR corresponds to
the French electricity market, containing hourly prices with associated load and generation forecasts
from 2012-01-09 to 2017-12-31. (5) DE represents the German electricity market, providing hourly
prices, zonal load forecasts in the TSO Amprion zone, and wind and solar generation forecasts from
2012-01-09 to 2017-12-31. Further statistics are reported in Table 6.


To assess CoRA’s capability in multi-modal covariate-aware forecasting, we employ RT-1 ( **?** ), a
large-scale robotic dataset with about 130k demonstrations collected over 17 months using 13 robots
in office kitchen environments. It covers 744 skills, ranging from basic object manipulation to longhorizon instructions, each paired with natural language commands and visual observations. The
dataset provides rich multi-modal supervision, supporting studies on instruction-conditioned and
multi-modal forecasting. The RT-1 dataset is particularly valuable for studying multi-modal and
instruction-conditioned forecasting, as it provides paired visual observations and natural language
descriptions aligned with robotic trajectories. In our experiments, we use a subset of RT-1, specifically the ’Move Object Near Object’ skill, and further restrict it to series with lengths no shorter
than 45. Each sequence is partitioned into training, validation, and test sets by assigning the last four
points as test targets and the preceding four points as validation targets, with the remaining points
used for training. This protocol guarantees at least one validation and one test instance per series, under a setup with an input length of 32 and a prediction horizon of 4. Time-MMD ( **?** ) is a large-scale
multi-modal dataset encompassing nine diverse domains, including agriculture, climate, healthcare,
and transportation. Each time series is paired with corresponding textual information sourced from
curated domain reports and structured web search results, enabling evaluation of text-enhanced forecasting performance. For consistency with prior work (Han et al., 2025), we exclude the Agriculture
and Economy subsets, and keep all other experimental settings identical to the official configuration.
Details of these datasets are provided in Table 7.


14


Table 6: Detailed dataset descriptions. _Nums_ denotes the number of covariates. _Freq_ denotes the
sampling interval of time points. The dataset size is given as (Train, Validation, Test).


Dataset Domain Nums Freq Target Variate Covariate Dataset Size Prediction Horizon


Electricity Energy 320 1H Electricity Electricity (18317, 2633, 5261) (96, 192, 336, 720)
Consumption Consumption


Weather Weather 20 10M CO2 Concentration Climate Feature (36792, 5271, 10540) (96, 192, 336, 720)


ETTh Energy 6 1H Oil Temperature Power Load Feature (8545, 2881, 2881) (96, 192, 336, 720)


ETTm Energy 6 15M Oil Temperature Power Load Feature (34465, 11521, 11521) (96, 192, 336, 720)


Traffic Traffic 861 1H Road Occupancy Rates Road Occupancy Rates (12185, 1757, 3509) (96, 192, 336, 720)


NP Electricity 2 1H Nord Pool Electricity Grid Load, Wind (36500, 5219, 10460) 24
Price Power


PJM Electricity 2 1H PJM Electricity Price System Load, Zonal (36500, 5219, 10460) 24
COMED Load


BE Electricity 2 1H Belgium Electricity Generation, System (36500, 5219, 10460) 24
Price Load


FR Electricity 2 1H France Electricity Price Generation, System (36500, 5219, 10460) 24
Load


DE Electricity 2 1H German Electricity Wind Power, Amprion (36500, 5219, 10460) 24
Price Zonal Load


Table 7: Detailed descriptions of RT-1 (Brohan et al., 2022) and TimeMMD ( **?** ).


Dataset Domain Num. Obs. Num. Series Freq Target Variate Covariate Type Prediction Horizon


RT-1 Solar Power 33,420 2871 13 [S] height to bottom Image 4


Agriculture 486 1 1M Retail Broiler Composite Text 12


Climate 496 1 1M Drought Level Text 12


Economy 423 1 1M International Trade Balance Text 12


Energy 1479 1 1M Gasoline Prices Text 12


A.2 BASELINE MODELS


We compared our method to multiple advanced baselines across various forecasting tasks.


**Time** **Series** **Foundation** **Models** We evaluate CoRA across multiple Time Series Foundation
Models, including Sundial (Liu et al., 2025), TimesFM (Das et al., 2023b), Chronos-Bolt (Ansari
et al., 2024), and FlowState (Graf et al., 2025). Specifically, on the Time-MMD dataset ( **?** ), we
further include Moirai (Liu et al., 2024c) and TabPFN-TS (Hoo et al., 2025) as baselines.


**Covariate-Aware Deep models** We compare CoRA with diverse advanced supervised deep forecasters. These include Transformer-based architectures such as TimeXer (Wang et al., 2024), iTransformer (Liu et al., 2023), PatchTST (Nie et al., 2022), Crossformer (Zhang & Yan, 2022), Autoformer (Wu et al., 2021), TiDE (Das et al., 2023a), Time-LLM (Jin et al., 2023), TTM (Ekambaram
et al., 2024) and TFT (Lim et al., 2021); classical sequence models such as N-BEATS (Oreshkin
et al., 2019), NBEATSx (Olivares et al., 2023) and DeepAR (Salinas et al., 2020); and other strong
baselines including DLinear (Zeng et al., 2023) and SCINet (Liu et al., 2022).


15


TimeMMD


Environment 11102 1 1M Air Quality Index Text 12


Health 1389 1 1W Influenza Patients Proportion Text 12


Security 297 1 1D Disaster and Emergency Grants Text 12


Social Good 900 1 1M Unemployment Rate Text 12


Traffic 531 1 1M Travel Volume Text 12


**Adaptation** **Method** We evaluate CoRA against other covariate adaptation methods, including
UniCA (Han et al., 2025), ChronosX (Arango et al., 2025), and AdaPTS (Benechehab et al., 2025).
In addition, to assess the role of covariates explicitly, we also compare with LoRA (Hu et al., 2021)
and SFT, which adapts model parameters without leveraging covariate signals.


A.3 IMPLEMENTATION DETAILS


All experiments are conducted using PyTorch on NVIDIA A100 Tensor Core GPUs. We employ the
Adam optimizer, along with the respective loss function of each foundation model, for optimization;
unless otherwise specified, the default loss function is mean squared error (MSE).


The training process is limited to a maximum of 50 epochs with early stopping, and patience is set
to 3. The learning rate is selected from the set _{_ 5e-6, 1e-5, 2e-5 _}_, and the batch size is fixed at 128.


For EPF, we follow the benchmark results reported in (Wang et al., 2024). For Time-MMD ( **?** ), we
use the results reported in (Han et al., 2025), both of which are strictly based on the configurations
in original papers. For all other results, we reproduce both the adaptation methods and the deep forecasting models from their official repositories, keeping hyperparameters and training configurations
unchanged to ensure a fair evaluation of each base model.


7: **E** [ˆ] = Concat - **E** ˆ [ts] _,_ ˆ **E** [txt] _,_ ˆ **E** [img][�]

8: **H** = Softmax( **W** CE) _·_ **E** [ˆ]
9: _γ, β, α_ = MLP - **H** 
10: **x** ˆ _T_ +1: _T_ + _H_ = (1 + _α_ ) TSFM-Head - _γ_ + (1 + _β_ ) **E** [˜] [target][�]

11: **return** **x** ˆ _T_ +1: _T_ + _H_


**Algorithm 2** Granger Causality Algorithm
**Require:** covariate series _A_, target series _B_, maximum lag _L_ max, criterion
**Ensure:** Granger causality strength _CE_, selected lag _l_

1: Select lag _l_ by minimizing criterion over 1 _, . . ., L_ max
2: Fit restricted model on _Bt_ _▷_ use _{Bt−_ 1 _, . . ., Bt−l}_, residual variance _σr_ [2]
3: Fit unrestricted model on _Bt_ _▷_ use _{Bt−_ 1 _, . . ., Bt−l, At−_ 1 _, . . ., At−l}_, residual variance _σu_ [2]

_r_
4: Compute Granger causality strength: _CE_ _←_ log _σ_ _[σ]_ _u_ [2][2]
5: **return** CE


B FULL RESULTS


B.1 FULL RESULTS OF UNI-MODAL COVARIATE-AWARE FORECASTING


Table 8 reports the complete results of the uni-modal covariate-aware forecasting task across widely
used datasets. All adaptation methods built on Sundial are fine-tuned only for the output horizon
of 720, consistent with the available pre-trained Sundial weights. For shorter horizons, the outputs
are obtained by truncating the 720-length predictions. In contrast, the baseline deep models are


16


**Algorithm 1** CoRA Algorithm
**Require:** Past target series **x** 1: _T_ = _{x_ 1 _, . . ., xT }_ ; Covariates **C** 1: _τ_ = _{_ **C** 1 _, . . .,_ **C** _τ_ _}_ (time series,
text, image); Prediction horizon _H_
1: **E** _[m]_ 1: _τ_ _[i]_ _i_ [= FM-Backbone(] **[C]** 1: _[m]_ _τ_ _[i]_ _i_ [)] _[,]_ _[i]_ [ = 1] _[, . . ., N,]_ _[m][i]_ _[∈{]_ [ts] _[,]_ [ txt] _[,]_ [ img] _[}]_


2: **E** [˜] [ts] = **E** [ts] _τ_
3: **E** [˜] [txt] = [1]


_τ_ [1] - _τt_ =1 **[E]** _t_ [txt] _[,]_ **[E]** [˜] [img] [=] _τ_ [1]


3: **E** [˜] [txt] = _τ_ [1] - _τt_ =1 **[E]** _t_ [txt] _[,]_ **[E]** [˜] [img] [=] _τ_ [1] - _τt_ =1 **[E]** _t_ [img]

4: **E** [target] 1: _T_ = TSFM-Backbone( **x** 1: _T_ )
5: **E** [˜] [target] = **E** [target] _T_
6: **E** [ˆ] _[m][i]_ = **E** [˜] _[m][i]_ **W** _[m][i]_ + **b** _[m][i]_ _,_ _i_ = 1 _, . . ., N,_ _mi_ _∈{_ ts _,_ txt _,_ img _}_


individually trained for each prediction length. Overall, adaptation methods on top of TSFMs consistently outperform conventional deep models, and our proposed CoRA achieves state-of-the-art
results, demonstrating its effectiveness as a general approach for covariate-aware adaptation.


Table 8: Full results of the long-term covariate-aware forecasting task. For all baselines, the lookback length _L_ is fixed at 2880 and dash (-) denotes out of memory (OOM) problem.


**CoRA** AdaPTS ChronosX UniCA TimeXer iTransformer PatchTST NBEATSx Crossformer DLinear
Models
**(Ours)** (2025) (2025) (2025) (2024) (2023) (2022) (2023) (2023) (2023)

|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|MSE MAE|
|---|---|---|---|---|---|---|---|---|
|**0.051 0.171** <br>**0.064 0.197** <br>**0.071 0.210** <br>**0.086 0.233**|0.054 0.174|0.066 0.195 <br> 0.075 0.213 <br> 0.086 0.232 <br> 0.113 0.268|0.055 0.174 <br> 0.070 0.201 <br> 0.085 0.227 <br> 0.128 0.286|0.078 0.227 <br> 0.084 0.235 <br> 0.090 0.244 <br> 0.102 0.255|0.075 0.219 <br> 0.114 0.270 <br> 0.160 0.324 <br> 0.292 0.455|0.080 0.229 <br> 0.084 0.235 <br> 0.088 0.239 <br> 0.130 0.291|0.153 0.326 <br> 0.176 0.349 <br> 0.206 0.383 <br> 0.190 0.347|0.167 0.340 <br> 0.299 0.463 <br> 0.500 0.565 <br> 0.576 0.635|
|**0.051 0.171** <br>**0.064 0.197** <br>**0.071 0.210** <br>**0.086 0.233**|<br> 0.068 0.199|<br> 0.068 0.199|<br> 0.068 0.199|<br> 0.068 0.199|<br> 0.068 0.199|<br> 0.068 0.199|<br> 0.068 0.199|<br> 0.068 0.199|
|**0.051 0.171** <br>**0.064 0.197** <br>**0.071 0.210** <br>**0.086 0.233**|<br> 0.079 0.220|<br> 0.079 0.220|<br> 0.079 0.220|<br> 0.079 0.220|<br> 0.079 0.220|<br> 0.079 0.220|<br> 0.079 0.220|<br> 0.079 0.220|
|**0.051 0.171** <br>**0.064 0.197** <br>**0.071 0.210** <br>**0.086 0.233**|<br> 0.101 0.252|<br> 0.101 0.252|<br> 0.101 0.252|<br> 0.101 0.252|<br> 0.101 0.252|<br> 0.101 0.252|<br> 0.101 0.252|<br> 0.101 0.252|


Avg **0.068** **0.203** 0.076 0.211 0.085 0.227 0.085 0.222 0.089 0.240 0.160 0.317 0.096 0.249 0.181 0.351 0.386 0.501 0.263 0.408


96 **0.111** **0.258** 0.112 0.256 0.258 0.389 0.125 0.272 0.168 0.329 0.175 0.339 0.188 0.349 0.245 0.407 0.270 0.410 0.250 0.402
192 **0.136** **0.291** 0.143 0.297 0.309 0.429 0.165 0.321 0.186 0.348 0.214 0.381 0.184 0.346 0.176 0.349 0.348 0.481 0.317 0.453
336 **0.149** **0.311** 0.157 0.317 0.353 0.462 0.199 0.359 0.192 0.355 0.304 0.455 0.190 0.348 0.206 0.383 0.383 0.509 0.323 0.460
720 **0.169** **0.335** 0.213 0.372 0.538 0.585 0.298 0.447 0.231 0.386 0.536 0.606 0.203 0.366 0.190 0.347 0.578 0.607 0.388 0.500


Avg **0.141** **0.299** 0.156 0.311 0.365 0.466 0.197 0.350 0.194 0.355 0.307 0.445 0.191 0.352 0.181 0.351 0.395 0.502 0.320 0.454


96 **0.026** **0.122** 0.027 0.123 0.028 0.124 0.030 0.128 0.038 0.147 0.036 0.148 0.031 0.134 0.066 0.199 0.038 0.154 0.030 0.131
192 **0.039** **0.149** 0.041 0.156 0.044 0.157 0.045 0.158 0.062 0.194 0.053 0.178 0.049 0.172 0.076 0.226 0.055 0.185 0.052 0.176
336 **0.048** **0.165** 0.054 0.181 0.057 0.181 0.056 0.177 0.069 0.203 0.065 0.197 0.061 0.195 0.203 0.381 0.077 0.219 0.069 0.201
720 **0.058** **0.182** 0.063 0.198 0.067 0.199 0.068 0.202 0.080 0.225 0.080 0.221 0.077 0.224 0.102 0.264 0.102 0.271 0.086 0.226


Avg **0.043** **0.155** 0.046 0.165 0.049 0.165 0.050 0.166 0.062 0.192 0.059 0.186 0.055 0.181 0.112 0.268 0.068 0.207 0.059 0.184

|0.094 0.228<br>0.121 0.265<br>0.155 0.310|Col2|
|---|---|
|<br> 0.094 0.228 <br> 0.121 0.265 <br> 0.155 0.310|<br> 0.117 0.263|
|<br> 0.094 0.228 <br> 0.121 0.265 <br> 0.155 0.310|<br> 0.153 0.310|


Avg **0.100** **0.237** 0.107 0.245 0.106 0.246 0.122 0.265 0.161 0.304 0.149 0.304 0.131 0.278 0.222 0.384 0.208 0.366 0.123 0.266


96 **0.001** **0.020** **0.001** 0.021 **0.001** 0.028 **0.001** 0.027 0.002 0.031 0.002 0.033 0.002 0.034 0.008 0.076 0.003 0.040 0.007 0.072
192 **0.001** **0.025** **0.001** **0.025** 0.002 0.031 0.002 0.031 0.002 0.032 0.002 0.033 0.002 0.035 0.105 0.092 0.003 0.042 0.008 0.076
336 **0.002** **0.028** **0.002** **0.028** **0.002** 0.034 **0.002** 0.034 **0.002** 0.033 **0.002** 0.034 **0.002** 0.034 0.009 0.085 0.004 0.051 0.008 0.079
720 **0.002** **0.032** **0.002** **0.032** **0.002** 0.037 **0.002** 0.039 **0.002** 0.034 **0.002** 0.036 0.003 0.041 0.010 0.090 0.005 0.056 0.008 0.078


Avg **0.001** **0.026** 0.002 0.027 0.002 0.033 0.002 0.033 0.002 0.033 0.002 0.034 0.002 0.036 0.033 0.086 0.004 0.047 0.008 0.076

|0.173 0.295 0.199 0.315|Col2|
|---|---|
|<br> 0.226 0.340 <br> 0.249 0.367|<br> 0.226 0.340 <br> 0.249 0.367|
|<br> 0.226 0.340 <br> 0.249 0.367|<br> 0.232 0.355|


Avg **0.194** **0.314** 0.212 0.329 0.206 0.323 0.230 0.347 0.292 0.387 0.293 0.406 0.327 0.431 0.352 0.449 0.352 0.446 0.264 0.376


96 **0.101** **0.169**   -   -   -   - 0.109 0.185 0.149 0.250 0.124 0.210 0.146 0.245 0.187 0.291 0.164 0.259 0.164 0.270
192 **0.109** **0.179**   -   -   -   - 0.118 0.197 0.156 0.258 0.131 0.221 0.152 0.253 0.210 0.316 0.225 0.317 0.179 0.290
336 **0.111** **0.187**   -   -   -   - 0.121 0.204 0.154 0.258 0.136 0.232 0.152 0.255 0.224 0.333 0.297 0.375 0.190 0.308
720 **0.128** **0.208**   -   -   -   - 0.141 0.226 0.168 0.271 0.163 0.265 0.165 0.267 0.267 0.372 0.411 0.378 0.280 0.400


Avg **0.112** **0.186**   -   -   -   - 0.122 0.203 0.157 0.259 0.139 0.232 0.154 0.255 0.222 0.328 0.274 0.332 0.203 0.317


17


B.2 FULL RESULTS OF MULTI-MODAL COVARIATE-AWARE FORECASTING


Table 9 reports the full results on the Time-MMD benchmark. We employ the Qwen3Embedding (Zhang et al., 2025) as a backbone in CoRA to derive text embeddings. Compared
to Sundial (Liu et al., 2025) in the zero-shot setting and Unica (Han et al., 2025), CoRA consistently
achieves superior performance across both deterministic metrics (MSE, MAE) and probabilistic metrics (CRPS). This demonstrates that CoRA successfully captures meaningful interactions between
temporal dynamics and textual covariates. These results further highlight the strength of CoRA as a
general and powerful strategy for integrating multi-modal information into TSFMs.


Table 9: Full results of multi-modal covariate-aware forecasting task on TimeMMD dataset.

|Col1|Models|CoRA UniCA Sundial NBEATS PatchTST DeepAR TFT TiDE Time-LLM TTM Moirai TabPFN-TS<br>(Ours) (2025) (2025) (2023) (2022) (2020) (2021) (2023a) (2023) (2024) (2024) (2025)|
|---|---|---|
|**Average**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|**0.641**<br>0.661<br>0.662<br>0.882<br>0.933<br>1.361<br>0.947<br>0.927<br>0.835<br>0.820<br>0.751<br>0.795<br>**0.580**<br>0.591<br>0.591<br>0.782<br>0.793<br>1.605<br>0.992<br>0.869<br>0.723<br>0.685<br>0.696<br>0.787<br>**0.690**<br>0.716<br>0.716<br>0.884<br>1.009<br>1.219<br>0.958<br>0.976<br>0.847<br>0.866<br>0.821<br>0.837<br>**0.653**<br>0.677<br>0.678<br>0.980<br>0.996<br>1.260<br>0.891<br>0.937<br>0.935<br>0.909<br>0.735<br>0.762|
|**Climate**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|0.536<br>0.567<br>0.567<br>0.668<br>0.724<br>0.737<br>0.695<br>0.575<br>0.634<br>0.526<br>0.596<br>**0.525**<br>0.440<br>0.487<br>0.487<br>0.519<br>0.640<br>0.623<br>0.599<br>0.465<br>0.468<br>0.408<br>0.488<br>**0.407**<br>**0.562**<br>0.595<br>0.595<br>0.712<br>0.788<br>0.779<br>0.768<br>0.685<br>0.687<br>0.635<br>0.706<br>0.638<br>0.607<br>0.620<br>0.620<br>0.773<br>0.743<br>0.809<br>0.719<br>0.574<br>0.746<br>0.535<br>0.593<br>**0.529**|
|**Energy**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|**0.888**<br>0.892<br>0.892<br>1.611<br>1.274<br>3.768<br>1.018<br>1.303<br>1.253<br>1.216<br>1.011<br>1.233<br>**0.838**<br>0.846<br>0.846<br>1.706<br>1.305<br>6.328<br>1.047<br>1.391<br>1.217<br>1.019<br>1.024<br>1.370<br>**0.928**<br>0.930<br>0.930<br>1.429<br>1.252<br>2.368<br>1.004<br>1.138<br>1.161<br>1.042<br>1.035<br>1.163<br>**0.897**<br>0.900<br>0.900<br>1.699<br>1.266<br>2.607<br>1.004<br>1.379<br>1.380<br>1.587<br>0.975<br>1.167|
|**Environment**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|**0.604**<br>0.608<br>0.608<br>0.725<br>0.644<br>0.689<br>0.638<br>0.638<br>0.699<br>0.644<br>0.641<br>0.644<br>0.527<br>**0.519**<br>**0.519**<br>0.628<br>0.589<br>0.648<br>0.601<br>0.572<br>0.617<br>0.546<br>0.623<br>0.611<br>**0.730**<br>0.742<br>0.742<br>0.809<br>0.785<br>0.822<br>0.763<br>0.778<br>0.774<br>0.777<br>0.756<br>0.772<br>0.554<br>0.564<br>0.564<br>0.739<br>0.558<br>0.596<br>0.550<br>0.564<br>0.707<br>0.609<br>**0.543**<br>0.550|
|**Health**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|**0.609**<br>0.637<br>0.637<br>0.873<br>0.930<br>1.131<br>1.014<br>0.973<br>0.862<br>0.966<br>0.776<br>0.969<br>**0.487**<br>0.514<br>0.513<br>0.739<br>0.874<br>1.023<br>1.059<br>0.916<br>0.735<br>0.906<br>0.722<br>0.964<br>**0.687**<br>0.706<br>0.706<br>0.860<br>0.928<br>1.118<br>1.004<br>0.992<br>0.846<br>0.989<br>0.821<br>1.008<br>**0.653**<br>0.692<br>0.692<br>1.020<br>0.989<br>1.251<br>0.979<br>1.010<br>1.004<br>1.002<br>0.786<br>0.936|
|**Security**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|**0.657**<br>0.688<br>0.689<br>0.847<br>1.170<br>1.419<br>1.399<br>1.521<br>0.862<br>0.763<br>0.746<br>0.678<br>**0.595**<br>0.620<br>0.620<br>0.692<br>0.882<br>1.078<br>1.614<br>1.260<br>0.690<br>0.676<br>0.669<br>0.612<br>**0.736**<br>0.763<br>0.764<br>0.927<br>1.332<br>1.607<br>1.409<br>1.767<br>0.951<br>0.880<br>0.856<br>0.764<br>**0.641**<br>0.682<br>0.683<br>0.922<br>1.295<br>1.571<br>1.175<br>1.535<br>0.946<br>0.732<br>0.714<br>0.657|
|**SocialGood**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|**0.745**<br>0.778<br>0.778<br>0.863<br>1.219<br>1.386<br>1.264<br>0.952<br>1.052<br>0.980<br>0.781<br>0.903<br>0.784<br>0.762<br>0.762<br>0.780<br>0.877<br>1.231<br>1.469<br>0.973<br>0.932<br>0.816<br>**0.735**<br>0.917<br>**0.719**<br>0.788<br>0.788<br>0.843<br>1.347<br>1.403<br>1.172<br>0.943<br>1.036<br>1.062<br>0.803<br>0.912<br>**0.733**<br>0.784<br>0.785<br>0.967<br>1.434<br>1.523<br>1.150<br>0.941<br>1.188<br>1.061<br>0.804<br>0.881|
|**Traffc**|**Average**<br>**MSE**<br>**MAE**<br>**CRPS**|0.448<br>0.458<br>0.458<br>0.584<br>0.569<br>**0.401**<br>0.599<br>0.529<br>0.484<br>0.647<br>0.704<br>0.616<br>0.390<br>0.387<br>0.387<br>0.408<br>0.385<br>**0.305**<br>0.552<br>0.506<br>0.401<br>0.428<br>0.610<br>0.631<br>0.470<br>0.488<br>0.488<br>0.608<br>0.632<br>**0.435**<br>0.589<br>0.528<br>0.475<br>0.679<br>0.772<br>0.605<br>0.484<br>0.498<br>0.498<br>0.737<br>0.689<br>**0.462**<br>0.657<br>0.553<br>0.576<br>0.834<br>0.731<br>0.611|


B.3 FULL RESULTS OF MULTIVARIATE FORECASTING


Table 10 summarizes results of multivariate forecasting across seven widely used datasets. On this
benchmark, CoRA achieves state-of-the-art performance across all datasets, substantially improving
upon recent deep forecasters. These results demonstrate that CoRA can jointly predict multiple
target variables in a unified manner, highlighting its effectiveness as a general adaptation strategy.


B.4 FULL RESULTS OF GENERALITY


We conduct extensive experiments on the EPF dataset using several representative TSFMs. As
shown in Table 11, CoRA consistently improves the performance of all TSFMs across both MSE and
MAE metrics. Compared with their zero-shot baselines, the improvements are significant, demonstrating the generality and effectiveness of CoRA as a universal covariate adaptation method. We
report results under the same training configuration and additionally provide the relative improvement ratio in MSE as a more intuitive assessment of the benefits brought by CoRA.


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


Table 10: Full results of the multivariate forecasting task. For all baselines, the look-back length _L_
is fixed at 2880, and _Avg_ means the average results from all four prediction lengths.


**CoRA** Timer-XL TimeXer iTransformer PatchTST Crossformer TiDE DLinear SCINet Autoformer
Models
**(Ours)** (2024d) (2024) (2023) (2022) (2023) (2023a) (2023) (2022) (2021)


Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE


96 **0.344** **0.381** 0.483 0.485 0.411 0.438 0.436 0.466 0.428 0.450 0.479 0.494 0.565 0.536 0.433 0.451 0.713 0.625 0.630 0.524
192 **0.387** **0.408** 0.520 0.506 0.442 0.459 0.469 0.487 0.476 0.477 0.587 0.550 0.634 0.572 0.479 0.482 0.736 0.638 0.762 0.519
336 **0.412** **0.425** 0.540 0.564 0.477 0.484 0.510 0.515 0.519 0.504 0.641 0.600 0.672 0.593 0.533 0.519 0.773 0.658 0.886 0.766
720 **0.471** **0.473** 0.647 0.633 0.639 0.572 0.619 0.591 0.639 0.586 0.867 0.732 0.751 0.646 0.633 0.596 0.897 0.717 0.971 0.836


Avg **0.404** **0.422** 0.548 0.547 0.492 0.488 0.508 0.515 0.516 0.504 0.643 0.594 0.656 0.587 0.519 0.512 0.780 0.660 0.812 0.661


96 **0.271** **0.329** 0.314 0.378 0.350 0.409 0.344 0.414 0.369 0.426 0.725 0.622 0.442 0.468 0.458 0.474 0.544 0.535 0.731 0.635
192 **0.328** **0.373** 0.387 0.428 0.414 0.452 0.408 0.454 0.469 0.491 0.771 0.684 0.509 0.504 0.547 0.521 0.614 0.570 0.789 0.677
336 **0.353** **0.397** 0.445 0.473 0.455 0.482 0.473 0.502 0.563 0.548 0.852 0.701 0.582 0.549 0.667 0.619 0.669 0.596 0.898 0.738
720 **0.372** **0.424** 0.541 0.538 0.595 0.561 0.533 0.533 0.558 0.547 0.893 0.755 0.688 0.606 0.808 0.743 0.841 0.667 0.941 0.778


Avg **0.331** **0.381** 0.422 0.454 0.454 0.476 0.440 0.476 0.490 0.503 0.810 0.691 0.555 0.532 0.620 0.589 0.667 0.592 0.840 0.707


96 **0.294** **0.336** 0.313 0.374 0.356 0.397 0.342 0.388 0.333 0.382 0.330 0.384 0.317 0.364 0.311 0.358 0.391 0.427 0.740 0.627
192 **0.325** **0.361** 0.358 0.403 0.388 0.416 0.363 0.402 0.375 0.408 0.363 0.403 0.352 0.387 0.341 0.377 0.410 0.438 0.858 0.696
336 **0.347** **0.380** 0.397 0.430 0.403 0.429 0.386 0.419 0.442 0.453 0.413 0.445 0.371 0.397 0.366 0.392 0.431 0.450 0.895 0.705
720 **0.381** **0.407** 0.456 0.469 0.444 0.455 0.423 0.444 0.449 0.453 0.639 0.596 0.413 0.422 0.410 0.422 0.468 0.472 0.934 0.700


Avg **0.337** **0.371** 0.381 0.419 0.398 0.424 0.379 0.413 0.400 0.424 0.436 0.457 0.363 0.393 0.357 0.387 0.425 0.447 0.857 0.682


96 0.167 **0.252** 0.225 0.319 0.189 0.287 0.189 0.285 0.182 0.280 0.391 0.469 0.198 0.294 **0.165** 0.262 0.242 0.337 0.381 0.453
192 0.224 **0.295** 0.291 0.366 0.249 0.330 0.238 0.318 0.238 0.317 0.475 0.514 0.323 0.387 **0.220** 0.304 0.282 0.361 0.449 0.493
336 0.278 **0.334** 0.344 0.402 0.291 0.352 0.298 0.356 0.311 0.368 0.663 0.674 0.332 0.390 **0.268** 0.338 0.322 0.387 0.503 0.521
720 **0.354** **0.388** 0.412 0.445 0.368 0.402 0.377 0.407 0.437 0.454 0.745 0.716 0.372 0.409 0.410 0.435 0.385 0.426 0.494 0.514


Avg **0.256** **0.317** 0.318 0.383 0.274 0.343 0.276 0.342 0.292 0.355 0.569 0.593 0.306 0.370 0.266 0.335 0.308 0.378 0.457 0.495


96 **0.158** **0.206** 0.255 0.299 0.186 0.246 0.187 0.252 0.160 0.219 0.159 0.212 0.171 0.231 0.169 0.230 0.168 0.231 0.400 0.433
192 0.201 **0.248** 0.315 0.344 0.233 0.286 0.231 0.291 0.210 0.265 **0.198** 0.263 0.211 0.265 0.210 0.267 0.216 0.274 0.447 0.448
336 0.249 **0.288** 0.331 0.366 0.281 0.318 0.273 0.325 0.273 0.309 **0.246** 0.298 0.253 0.296 0.257 0.310 0.299 0.333 0.462 0.452
720 0.311 0.333 0.361 0.384 0.347 0.361 0.314 0.352 0.359 0.368 0.335 0.369 **0.300** **0.332** 0.314 0.357 0.314 0.344 0.693 0.616


Avg **0.230** **0.269** 0.316 0.348 0.262 0.303 0.251 0.305 0.251 0.290 0.235 0.285 0.234 0.281 0.237 0.291 0.249 0.296 0.500 0.487


96 **0.124** **0.220** 0.131 0.229 0.137 0.241 0.167 0.275 0.136 0.240 0.133 0.232 0.130 0.226 0.129 0.227 0.144 0.252 0.256 0.362
192 **0.142** **0.238** 0.147 0.244 0.154 0.256 0.177 0.283 0.151 0.254 0.162 0.266 0.146 0.242 0.144 0.242 0.163 0.271 0.267 0.371
336 **0.159** **0.256** **0.159** 0.257 0.189 0.291 0.196 0.302 0.167 0.269 0.191 0.286 0.163 0.259 **0.159** 0.260 0.178 0.286 0.278 0.376
720 0.194 0.287 **0.183** **0.279** 0.210 0.311 0.234 0.335 0.199 0.297 0.249 0.338 0.199 0.290 0.192 0.292 0.239 0.331 0.367 0.451


Avg **0.155** **0.250** **0.155** 0.252 0.172 0.275 0.194 0.299 0.163 0.265 0.184 0.281 0.160 0.254 0.156 0.255 0.181 0.285 0.292 0.390


96 **0.350** **0.245** 0.569 0.428 0.377 0.269 0.375 0.275 0.397 0.286 0.481 0.256 0.377 0.264 0.379 0.270 0.455 0.342 0.538 0.405
192 **0.372** **0.257** 0.570 0.513 0.387 0.274 0.395 0.284 0.410 0.293 0.492 0.270 0.390 0.269 0.392 0.276 0.462 0.346 0.776 0.468
336 **0.389** **0.267** 0.589 0.521 0.400 0.281 0.410 0.292 0.423 0.299 0.514 0.277 0.403 0.275 0.407 0.284 0.481 0.356 0.769 0.460
720 **0.426** **0.288** 0.658 0.577 0.440 0.298 0.447 0.312 0.457 0.313 0.601 0.337 0.438 0.294 0.447 0.307 0.512 0.363 0.885 0.524


Avg **0.384** **0.265** 0.597 0.510 0.401 0.281 0.407 0.291 0.422 0.298 0.522 0.285 0.402 0.276 0.406 0.284 0.478 0.352 0.742 0.464


Table 11: Full results of CoRA generalize to other Time Series Foundation Models.


Datasets NP PJM BE FR DE Avg


Models MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE

|0.263 0.288<br>0.222 0.246|0.089 0.186<br>0.073 0.165|0.364 0.271<br>0.339 0.236|0.361 0.217<br>0.357 0.206|0.543 0.462<br>0.401 0.388|
|---|---|---|---|---|
|15.59%|17.98%|6.87%|1.11%|26.15%|
|0.255<br>**0.271**<br>**0.246**<br>**0.271**|0.085<br>**0.182**<br>**0.083**<br>**0.182**|0.383<br>0.252<br>**0.380**<br>**0.251**|0.398<br>0.206<br>**0.394**<br>**0.205**|0.526<br>0.456<br>**0.487**<br>**0.433**|
|3.53%|2.35%|0.78%|1.01%|7.41%|
|0.246<br>0.265<br>**0.235**<br>**0.255**|0.082<br>0.178<br>**0.076**<br>**0.170**|0.356<br>0.239<br>**0.353**<br>**0.233**|0.357<br>0.191<br>**0.352**<br>**0.184**|0.494<br>0.442<br>**0.445**<br>**0.414**|
|4.47%|7.32%|0.84%|1.40%|9.92%|
|0.229<br>0.256<br>**0.225**<br>**0.253**|0.081<br>0.<br>**0.078**<br>**0.177**|0.362<br>0.252<br>**0.355**<br>**0.243**|0.365<br>0.203<br>**0.364**<br>**0.199**|0.497<br>0.446<br>**0.464**<br>**0.424**|
|1.75%|3.70%|1.93%|0.27%|6.64%|


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


C SHOWCASES


To facilitate a clear comparison among various models, we present additional prediction showcases
for uni-modal covariate-aware forecasting in Figure 8. These examples are provided by the following
methods: AdaPTS (Benechehab et al., 2025), TimeXer (Wang et al., 2024), and PatchTST (Nie et al.,
2022). Of all the models, CoRA delivers the most accurate future series predictions. Additionally,
we provide the showcases of multi-modal covariate-aware forecasting in Figure 9.


**CoRA** **AdaPTS** **TimeXer** **PatchTST**


Figure 8: Visualization of uni-modal covariate-aware results on NP, DE and ECL dataset.


D LIMITATIONS


A notable limitation of CoRA lies in its treatment of temporally aligned auxiliary modalities such
as language and image sequences. At present, CoRA applies a simple mean aggregation along the
temporal dimension, which inevitably discards fine-grained temporal dynamics and leads to underutilization of the rich and potentially complementary information contained in these modalities.
Future work could investigate more sophisticated fusion strategies that explicitly preserve temporal
dependencies, thereby enabling CoRA to more effectively leverage auxiliary modalities and further
improve its adaptability across diverse forecasting scenarios.


20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


Figure 9: Visualization of multi-modal covariate-aware results on RT-1 dataset.


21