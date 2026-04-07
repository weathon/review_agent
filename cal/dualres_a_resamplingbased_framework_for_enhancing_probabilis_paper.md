# DUALRES: A RESAMPLING-BASED FRAMEWORK FOR ENHANCING PROBABILISTIC FORECASTING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Probabilistic forecasting of time series has gained increasing attention in practice
due to the need for assessing risks and uncertainties in future observations. In
this manuscript, we propose DualRes, a framework that improves the probabilistic forecasting performance of existing algorithms by incorporating conditional
heteroskedasticity and residual distributional information. Specifically, during
training, DualRes employs two separate models to learn the conditional mean and
volatility of the time series, while during inference it generates pseudo-normalized
residuals through resampling. DualRes requires only mean forecasts, so it offers
substantial flexibility in the choice of forecasting algorithms—even algorithms
originally designed for mean forecasting can be adapted to probabilistic forecasting. DualRes applies to both univariate and multivariate time series and remains
robust under non-Gaussian errors with conditional heteroskedasticity. Numerical
experiments on six real-world datasets demonstrate its good empirical performance
in capturing distribution of future observations and producing accurate prediction
intervals.


1 INTRODUCTION


Time series is a common data type in real-world applications such as finance, energy management,
and weather forecasting. After collecting a sequence of time series data, this manuscript focuses on
probabilistic forecasting, which aims to predict the probability distribution of future observations
and thereby support risk assessing and decision-making, as discussed in Luo et al. (2018); Nguyen &
Quanz (2021); Wu & Politis (2024); Zheng et al. (2025) and the references therein.


To our knowledge, two types of methods are commonly considered in probabilistic forecasting. The
first type, such as the work of Kollovieh et al. (2023); Chen et al. (2024b;a); Tashiro et al. (2021);
Zheng et al. (2025), leveraged diffusion process and generative model, like those of Song et al.
(2020); Ho et al. (2020); Kollovieh et al. (2025), to perform probabilistic forecasting. The validity
of such methods in general relied on the assumption of time series having Gaussian distribution.
Another stream that addressed probabilistic forecasting problems involved adjusting the training
processes. Notable examples include Le Guen & Thome (2020); Rasul et al. (2021b); Hasson et al.
(2021); Bergsma et al. (2023); Ansari et al. (2024). A common issue of these methods is that the
underlying mathematical models and mechanisms of their validity are not transparent and rigorous to
practitioners compared to those of diffusion model-based approaches.


In this manuscript, motivated by recent advances in bootstrap and resampling methods for statistical
inference and prediction in time series analysis Wu & Politis (2024; 2025); Zhang et al. (2024),
we propose DualRes, a resampling-based framework for probabilistic forecasting of time series
data. DualRes consists of three steps. First, we train a predictive model—such as those in Zeng
et al. (2023); Lin et al. (2024)—to estimate the conditional mean of the time series, and compute
fitted residuals as the difference between the observations and the predictive means. Second, we
introduce another model to estimate the conditional volatility, and normalize the fitted residuals
by dividing them by the predicted volatilities. Finally, we apply bootstrap algorithms (see Efron
(1979)) to resample the normalized residuals, and combine the estimated conditional mean and
volatility to generate predictive distributions of future observations. As demonstrated in Wu (1986);
Stine (1985); Chwialkowski et al. (2014), a well-designed bootstrap algorithm can approximate the
underlying probability distribution of future time series without imposing restrictive distributional


1


assumptions, such as Gaussianity. Thus, DualRes relaxes the reliance on Gaussian distributions of
diffusion process-based methods.


In addition to relaxing the Gaussian assumption, DualRes offers several advantages. First, it is
flexible in the choice of conditional mean and volatility models. As shown Section 4.1, by applying a
logarithmic transformation to the squared residuals, DualRes requires only mean forecasts to perform
probabilistic forecasting. This allows models originally designed for mean forecasting to be adapted
for probabilistic forecasting. Second, DualRes explicitly accounts for conditional heteroskedasticity
and non-Gaussianity, thereby improving the performance of probabilistic forecasting methods that
ignore these features. Finally, as established in Theorem 1, DualRes incorporates spatial dependence
by resampling residual vectors, making it adaptable to multivariate time series settings.


We summarize the advantages of the proposed method as follows.


    - **No Gaussianity assumption:** Our work does not rely on maximizing likelihood functions,
so the data distributions are not necessarily Gaussian.

    - **Flexibility in selecting mean/volatility forecasting algorithms:** Implementation of our
work only needs models generating mean forecasts, thus offering good flexibilities.

    - **Theoretical justification:** The validity of our approach stems from its ability to simulate the
underlying data-generating process of time series instead of a black-box model. Furthermore,
under some conditions, the resampling mechanism is ensured to capture the underlying
distribution of innovations.

    - **Robustness** **to** **conditional** **heteroskedasticity** **and** **multivariate** **Settings:** DualRes is
adaptable for conditional heteroskedastic time series, and it accounts for spatial dependence
in predictions.


2 RELATED WORKS


This work is related to the area of probabilistic time series forecasting and resampling. We provide
a brief introduction of the latest studies for each area. In addition, we introduce the setting of
probabilistic forecasting to make the manuscript self-contained.


**Probabilistic time series forecasting.** Diffusion models and their variants, like those introduced in
Ho et al. (2020), have been applied to both univariate and multivariate probabilistic forecasting of time
series Rasul et al. (2021a;b); Li et al. (2022); Chen et al. (2024b;a); Kollovieh et al. (2025); Zheng et al.
(2025). By modeling time series data as a Markov chain with Gaussian transitions, these methods
offer good interpretability in the training and inference stage. The state space model is another
frequently used model that offers good interpretability and empirical performance. Recent works
such as Rangapuram et al. (2018); Li et al. (2019) leveraged deep learning to describe parameters
in the state space model. We also refer Rangapuram et al. (2021); Feng et al. (2024); Ansari et al.
(2024) for other deep learning-based approaches to probabilistic forecasting.


**Resampling and bootstrap.** Bootstrap algorithm is a well-recognized method to quantify uncertainty
of statistics, and has been employed to various fields of machine learning, like those in White &
White (2010); Austern & Syrgkanis (2021); Shin et al. (2021); Rohekar et al. (2018); Wang et al.
(2024b); Yu et al. (2024).


3 RESAMPLING ASSISTED PROBABILISTIC FORECASTING (DUALRES)


Suppose we observe a time series **x** 1: _T_ _∈_ **R** _[d]_ _,_ with _t_ = 1 _, · · ·_ _, T_ denoting the time steps. Our
objective is to forecast the distributions of future observations **x** _T_ + _j_ for _j_ = 1 _,_ 2 _, · · ·_ _, J_ . There
have been discussions in the literature like Salinas et al. (2020) and Kollovieh et al. (2025). When
further investigating these works, we find that they effectively incorporated the conditional mean
and conditional volatility information in forecasting. However, these works commonly assigned
a Gaussian distribution to the residuals, making the validity of forecasting algorithms rely on the
residuals (and therefore, observations) obeying Gaussian distributions.


Our objective is to take into account the distributional information and avoid the assumption of
Gaussian distribution in forecasting. To achieve the goal, we incorporate a resampling step into the


2


Figure 1: Structure of the training and inference stage.


forecasting algorithm 2. Resampling has been well employed in the literature such as Pan & Politis
(2016), Wu & Politis (2025), and Zhang et al. (2025) in forecasting. However, to our knowledge, they
did not account for the conditional heteroskedasticity (i.e., dependence of future variance on past
observations), while our work allows for the existence of conditional heteroskedasticity in future
observations.


3.1 TRAINING STAGE


Figure 1 presents an overview about the structure of the training and inference of stage of the proposed
method. Our work is motivated by a two-stage conditional heterogeneous vector autoregressive model


**x** _t_ = _F_ ( **x** _t−_ 1 _, · · ·_ _,_ **x** _t−q_ ) + _**ζ**_ _t,_ and _**ζ**_ _t_ = _G_ ( _**ζ**_ _t−_ 1 _, · · ·_ _,_ _**ζ**_ _t−s_ ) _**η**_ _t,_ (1)


where


_G_ ( _**ζ**_ _t−_ 1 _, · · ·_ _,_ _**ζ**_ _t−s_ ) = diag ( _G_ 1( _**ζ**_ _t−_ 1 _, · · ·_ _,_ _**ζ**_ _t−s_ ) _, · · ·_ _, Gd_ ( _**ζ**_ _t−_ 1 _, · · ·_ _,_ _**ζ**_ _t−s_ ))


is a _d × d_ diagonal matrix, _F_ : **R** _[d][×][q]_ _→_ **R** _[d]_ _, Gi_ : **R** _[d][×][s]_ _→_ [0 _, ∞_ ) are functions to learn, and _**η**_ _t_ are
independent of past observations **x** _−t_ and _**ζ**_ _−t_, **E** - _**η**_ [(] _[t]_ [)][�] = 0, and _**η**_ [(] _[t]_ [)] have identical distribution.


The functions _F_ and _G_ respectively controls the conditional mean and conditional volatility of time
series data, Furthermore, such model offers a good property that the residual terms _**ζ**_ _t_ does not incur
bias to the conditional mean _F,_ which motivates the two-stage training procedure as in Algorithm 1.
We prove this property in Section 4.


3


**Algorithm 1** Training a heterogeneous vector autoregressive model


**Require:** Time series data _{_ **x** _t_ : _t_ = 1 _, · · ·_ _, T_ _},_ lag _q_ for the conditional mean model, and lag _s_ for
the conditional volatility model.
1: Train the conditional mean model _F_ and derive the fitted residuals

[�]

_**ζ**_               - _t_ = **x** _t −_ _F_ �( **x** _t−q, · · ·_ _,_ **x** _t−_ 1)


for _t_ = _q_ + 1 _, · · ·_ _, T_ .
2: Train the conditional volatility model _G_ [�] with the fitted residuals _**ζ**_ [�] _t, t_ = _q_ + 1 _, · · ·_ _, T._ After that,
derive the normalize fitted residuals


                     _**η**_             - _t_ = _G_ [�] _[−]_ [1][ �] _**ζ**_             - _t−s, · · ·_             - _**ζ**_ _t−_ 1 _**ζ**_             - _t,_ (2)


where _t_ = _q_ + _s_ + 1 _, · · ·_ _, T._


**Remark** **1.** _Practitioners_ _may_ _resort_ _to_ _mean_ _forecasting_ _methods,_ _such_ _as_ _Lin_ _et_ _al._ _(2024),_ _to_
_establish the model_ _F_ _for the conditional mean function F_ _in equation 1._ _Learning G, on the other_

[�]
_hand,_ _is_ _not_ _straightforward._ _After_ _calculating_ _**ζ**_ [�] _t,_ _this_ _manuscript_ _performs_ _the_ _transformation_

- _**ι**_ _t_ = _R_ ( _**ζ**_ [�] _t_ ) _for t_ = _q_ + 1 _, · · ·_ _, T, where R_ : **R** _[d]_ _→_ **R** _[d]_ _is a function of the form:_


_R_ ( **x** ) = (log( **x** [2] 1 [)] _[,]_ [ log(] **[x]** [2] 2 [)] _[,][ · · ·]_ _[,]_ [ log(] **[x]** [2] _d_ [))] _[⊤]_ _and_ **x** _∈_ **R** _[d]_ _._ (3)


_We then use mean forecasting methods (e.g., those in Lin et al. (2024)) to learn Ui_ = log( _Gi_ ) _. We_
_demonstrate in Section 4.1 that, despite taking logarithm transformations incur a constant bias when_
_learning_ log( _Gi_ ) _, the constant bias will be self-eliminated during the normalization step equation 2_
_of Algorithm 1 and the sampling step equation 4 of the inference Algorithm 2._ _Consequently, the bias_
_introduced during the training stage does not affect the prediction._


The motivation of the model equation 1 originates from the ARMA-GARCH model, like those in Ling
& McAleer (2003), that adopted linear models for both _F_ and _G._ The conditional heteroskedasticity
considered in this manuscript associates the volatility with past observations, and is different from Ye
et al. (2025), where the volatility was associated with exogenous features.


The flexibility of Algorithm 1 is reflected by its selection of models used to learn _F_ and _G_ —mean
forecasting algorithms, such as those proposed in Zeng et al. (2023); Zhang & Yan (2023); Lin et al.
(2024), among others—can be employed to fulfill this purpose.


3.2 INFERENCE STAGE


The intuition behind Algorithm 2 involves simulating the data generating process in equation 1. If _F_

[�]
and _G_ closely approximate the true conditional mean _F_ and conditional volatilities _G,_ then Theorem 1

[�]
in Section 4 guarantees that the distribution of the simulated normalized residuals _**η**_ _j_ _[∗]_ [closely matches]
the distribution of the true normalized residuals _**η**_ _j_ . Furthermore, the generation of **x** _[∗]_ _T_ + _j_ [follows]
the same autoregressive iteration as in equation 1. Therefore, under the assumption that equation 1
accurately characterizes the data generating process of **x** _t,_ since the estimated conditional mean _F_ [�],
conditional volatility _G_ [�], the distribution of pseudo-normalized residuals _**η**_ _j_ _[∗]_ [, and the autoregressive]
iteration all provide good approximations to that of **x** _t_, the distribution of the pseudo-samples **x** _[∗]_ _T_ + _j_
should be close to that of the actual future observations **x** _T_ + _j_ .


4


**Algorithm 2** Inference Stage


**Require:** Time series data **x** 1: _T_, lag _q_ for conditional mean, lag _s_ for conditional volatility, prediction
step _J_, resampling time _B_ .
1: Derive the functions _F_ [�] and _G,_ [�] as well as the normalized fitted residuals _**η**_  - _t_ as in Algorithm 1.
2: **for** _b ←_ 1 to _B_ **do**
3: Sample _**η**_ _j_ _[∗]_ [for] _[ j]_ [= 1] _[,][ · · ·]_ _[, J]_ [by drawing from] _**[η]**_ [�] _[q]_ [+] _[s]_ [+1] _[,][ · · ·]_ _[,]_ [ �] _**[η]**_ _[T]_ [with replacement.]
4: Generate pseudo-samples **x** _[∗]_ _T_ +1 _[,][ · · ·]_ _[,]_ **[ x]** _[∗]_ _T_ + _j_ [using the following iteration:]

_**ζ**_ _T_ _[∗]_ + _j_ [=] _[G]_ [�][(] _**[ζ]**_ [�] _T_ _[∗]_ + _j−s_ _[,][ · · ·]_ _[,]_ [ �] _**[ζ]**_ _T_ _[∗]_ + _j−_ 1 [)] _**[η]**_ _j_ _[∗][,]_

(4)
**x** _[∗]_ _T_ + _j_ [=] _[F]_ [�][(] **[x]** _T_ _[∗]_ + _j−q_ _[,][ · · ·]_ _[,]_ **[ x]** _[∗]_ _T_ + _j−_ 1 [) +] _**[ ζ]**_ _T_ _[∗]_ + _j_ _[,]_

where **x** _[∗]_ _T_ + _j−q_ [=] **[ x]** _[T]_ [ +] _[j][−][q]_ [and][ �] _**[γ]**_ _T_ _[∗]_ + _j−s_ [=] _**[γ]**_ [�] _[T]_ [ +] _[j][−][s]_ [if] _[ q, s][ ≥]_ _[j.]_
5: **end for**
6: For any measurable set _A_ _⊂_ **R** _[d][×][J]_ _,_ we estimate the joint distribution of **x** ( _T_ +1):( _T_ + _J_ ) by the
empirical measure _B_ [1]   - _Bb_ =1 **[1][x]** _[∗]_ ( **T** + **1** ):( **T** + **J** ) _[∈]_ **[A]**


**Remark** **2.** _Practitioners_ _may_ _resort_ _to_ _Remark_ _1_ _to_ _learn_ _G._ _In_ _such_ _case,_ _the_ _value_ _of_
_G_ �( _**ζ**_ - _T_ _[∗]_ + _j−s_ _[,][ · · ·]_ _[,]_ [ �] _**[ζ]**_ _T_ _[∗]_ + _j−_ 1 [)] _[can]_ _[be]_ _[derived]_ _[through]_ _[applying]_ _[the]_ _[learned]_ _[autoregressive]_ _[model]_ _[to]_

- _**ι**_ _[∗]_ _T_ + _j−s_ _[,][ · · ·]_ _[,]_ [ �] _**[ι]**_ _[∗]_ _T_ + _j−_ 1 _[,][ where]_ [ �] _**[ι]**_ _[∗]_ _k_ [=] _[ R]_ [ (] _**[ζ]**_ _k_ _[∗]_ [)] _[ .]_


4 THEORETICAL JUSTIFICATION


The theoretical justification of DualRes is divided into two parts. First, we provide illustrations on
why Algorithm 1 is capable of learning _F_ and _G._ After that, we summarize in Theorem 1 that the
distribution of the pseudo-normalized residuals _**η**_ _j_ _[∗]_ [closely approximates that of the true normalized]
residuals _**η**_ _j._


4.1 FURTHER DISCUSSIONS ON SECTION 3


To illustrate why the two-stage procedure in Algorithm 1 learns _F_ and _G,_ from the tower property of
conditional expectation,


E - _**ζ**_ _t_ _|_ **x** ( _t−q_ ):( _t−_ 1)� = E �E - _G_ ( _**ζ**_ _t−_ 1 _, · · ·_ _,_ _**ζ**_ _t−s_ ) _**η**_ _t_ _|_ **x** ( _t−q_ ):( _t−_ 1) _,_ _**ζ**_ ( _t−s_ ):( _t−_ 1)� _|_ **x** ( _t−q_ ):( _t−_ 1)�

= E �( _G_ ( _**ζ**_ _t−_ 1 _, · · ·_ _,_ _**ζ**_ _t−s_ )E _**η**_ _t_ ) _|_ **x** ( _t−q_ ):( _t−_ 1)� = 0 _._


Therefore, when we train _F,_ [�] the residuals _**ζ**_ _t_ do not incur bias to _F,_ making it possible for the
estimator _F_ to closely approximate _F._ On the other hand, define the function _R_ as in equation 3,

[�]
define _**γ**_ _t_ = _R_ ( _**ζ**_ _t_ ) _,_ then the _i_ -th element of _**γ**_ _t_ is

_**γ**_ _t,i_ = log          - _G_ [2] _i_ [(] _**[ζ]**_ _[t][−]_ [1] _[,][ · · ·]_ _[,]_ _**[ ζ]**_ _[t][−][s]_ [)]          - + log          - _**η**_ _t,i_ [2]          - _._ (5)

Furthermore, by assuming that the functions _G_ [2] _i_ [(] _[·]_ [)] _[, i]_ [=] [1] _[,][ · · ·]_ _[, d,]_ [ depend on] _**[ ζ]**_ _[t][−]_ [1] _[,][ · · ·]_ _[,]_ _**[ ζ]**_ _[t][−][s]_ [only]
through their element-wise squares, and notice that _**ζ**_ _t,i_ [2] [= exp (] _**[γ]**_ _[t,i]_ [)] _[,]_ [ equation 5 implies that]


_**γ**_ _t_ = _A_ ( _**γ**_ _t−_ 1 _, · · ·_ _,_ _**γ**_ _t−s_ ) + _**ι**_ _t,_ (6)

where _A_ : **R** _[d][×][s]_ _→_ **R** _[d]_ is a function such that _Ai_ ( _**γ**_ _t−_ 1 _, · · ·_ _,_ _**γ**_ _t−s_ ) = log - _G_ [2] _i_ [(] _**[ζ]**_ _[t][−]_ [1] _[,][ · · ·]_ _[,]_ _**[ ζ]**_ _[t][−][s]_ [)] - +
E �log - _**η**_ _t,i_ [2] �� and _**ι**_ _t,i_ = log - _**η**_ _t,i_ [2] - _−_ E �log - _**η**_ _t,i_ [2] �� _._ Therefore, the representation equation 6 allows
the use of a mean-forecasting algorithm to learn _B,_ which inevitably incurs a constant bias term
E �log - _**η**_ _t,i_ [2] �� _._


Fortunately, the constant bias does not affect the prediction as it self-eliminated during equation 2
of Algorithm 1, which divides the fitted residuals _**ζ**_ [�] _t_ by _G,_ [�] and equation 4 of Algorithm 2, which
multiplies the sampled _**η**_ _j_ _[∗]_ [by] _[G.]_ [�]

We would like to stress that the assumption of _G_ [2] _i_ [depending] [on] _**[ζ]**_ _[t][−]_ [1] _[,][ · · ·]_ _[,]_ _**[ ζ]**_ _[t][−][s]_ [through] [their]
element-wise squares is common in the literature. For example, the ARMA-GARCH models in Ling


5


& McAleer (2003) leveraged this assumption. The advantage of this transformation is, by replacing _**γ**_ _t_
with � _**γ**_ _t_ = _R_ ( _**ζ**_ [�] _t_ ) _,_ - _**γ**_ _t_ approximately follows an additive autoregressive process equation 6, allowing
the use of various conditional mean forecasting methods—such as those in Lin et al. (2024)—for
estimating the function _A_ in equation 6.


4.2 VALIDITY OF THE RESAMPLE PROCEDURE


While conditional mean and volatility information has been widely leveraged in various probabilistic
forecasting algorithms, like Salinas et al. (2020); Zheng et al. (2025), the distributional information
of residuals _**η**_ _t_ has received comparatively less attention. Compared to directly assigning normal
distribution to _**η**_ _t,_ we introduce the resampling step equation 4 in Algorithm 2 to learn underlying
distribution of _**η**_ _t._


Furthermore, as illustrated in Section 3, the validity of Algorithm 2 comes from simulating the
underlying data generating process of **x** _t_ . Therefore, if model eq.equation 1 holds true and Algorithm
1 generates good estimators for _F_ and _G_ (up to a constant scale), the validity of Algorithm 2 is
achieved provided that the empirical process of the vector _**η**_ - _t_ —characterized by the probability
measure defined by the following joint cumulative distribution function (CDF in abbreviation)


where **1** _**η**_ �t _≤_ **y** denotes for [�] _i_ _[d]_ =1 **[1]** _**[η]**_ [�] t _,_ i _[≤]_ **[y]** i [, converges to the distributions of] _**[ η]**_ [(] _[t]_ [)] _[.]_ [ Theorem 1 provides a]
theoretical justification for this claim.
**Theorem** **1.** _Suppose_ _**η**_ _t, t_ = 1 _,_ 2 _, · · ·_ _,_ _are_ _independent_ _and_ _identical_ _distributed._ _In_ _addition,_
_suppose conditions detailed in Section A of Appendix hold true._ _Then we have_


sup [0] _[,]_ (8)
_**y**_ _∈_ **R** _[d][ |][P]_ [ �][(] _**[y]**_ [)] _[ −]_ _[P]_ [(] _**[y]**_ [)] _[| →][p]_


_where →p denotes convergence in probability, P_ ( _·_ ) _denotes the CDF of_ _**η**_ _t, and the convergence is_
_with respect to the sample size T_ _→∞._


_Proof._ Postponed to Section A in Appendix.


Theorem 1 guarantees that the distribution of the resampled normalized residuals _**η**_ _t,i_ _[∗]_ [in Algorithm 2]
matches that of the true normalized residuals _**η**_ _t,i_ _[∗]_ [.] [As a result, Algorithm 2 effectively captures the]
distributional information of _**η**_ _t,i_ _[∗]_ [.]

**Remark 3.** _According to Politis et al. (1999), sampling with replacement from_ _**η**_ - _t is equivalent to_
_drawing from the distribution with CDF_ _P_ ( _·_ ) _as defined in e.q._ _equation 7._ _Therefore, the distribution_

[�]
_of_ _**η**_ _i_ _[∗]_ _[is guaranteed to match the distribution of]_ _**[ η]**_ _[i]_ _[once e.q.]_ _[equation 8 is satisfied.]_


5 NUMERICAL EXPERIMENTS


This section demonstrates the effectiveness of DualRes as a boosting algorithm for enhancing the
performance of existing methods in both univariate and multivariate probabilistic forecasting. Due to
the space limitations, the detailed experimental setup and additional experimental results—including
hyperparameter choices, introduction of datasets and evaluation metrics, and demonstration of mean
forecasting performance—are deferred to Section B of the Appendix.


5.1 UNIVARIATE PROBABILISTIC FORECASTING


**Dataset and experimental settings.** We run the experiments on six real-world commonly used time
series dataset, respectively named _ETTh1, ETTh2, Electricity, Traffic, Exchange, and M4-Hourly._ The
details about these datasets are introduced in Section B.1 of the Appendix.


The evaluation metrics are CRPS and MAEC (mean absolute error of coverage). A detailed introduction to these metrics is provided in Section B.2 of the Appendix. In addition to probabilistic


6


1
_P_ �( **y** ) = _T_ _−_ _q −_ _s_


_T_

 - **1** _**η**_ �t _≤_ **y** (7)

_t_ = _s_ + _q_ +1


Table 1: Numerical experiment results on univariate time series datasets. The numbers in brackets
indicate 95% confidence intervals, computed from five independent repetitions of each experiment.
In the ablation studies, the better result is highlighted in bold, corresponding to smaller metric values,
or, when metrics are equal, to narrower confidence intervals.


Models Metrics ETTh1 ETTh2 Electricity Traffic Exchange M4-Hourly


DeepAR CRPS 0 _._ 178(0 _._ 031) **0** _._ ( **0** _._ ) 0 _._ 082(0 _._ 001) **0** _._ ( **0** _._ ) 0 _._ 015(0 _._ 001) 0 _._ 087(0 _._ 092)
MAEC 0 _._ 411(0 _._ 082) 0 _._ 394(0 _._ 148) 0 _._ 454(0 _._ 001) **0** _._ ( **0** _._ ) 0 _._ 498(0 _._ 003) 0 _._ 411(0 _._ 099)


DeepAR CRPS **0** _._ ( **0** _._ ) 0 _._ 085(0 _._ 002) **0** _._ ( **0** _._ ) 0 _._ 115(0 _._ 003) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )
+Ours MAEC **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 471(0 _._ 013) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )


DLinear CRPS **0** _._ ( **0** _._ ) 0 _._ 075(0 _._ 003) 0 _._ 061(0 _._ 007) **0** _._ ( **0** _._ ) 0 _._ 019(0 _._ 008) 0 _._ 048(0 _._ 005)
MAEC 0 _._ 414(0 _._ 014) 0 _._ 462(0 _._ 018) 0 _._ 382(0 _._ 016) 0 _._ 433(0 _._ 012) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )

DLinear CRPS 0 _._ 196(0 _._ 008) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 133(0 _._ 002) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )
+Ours MAEC **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 465(0 _._ 011) 0 _._ 409(0 _._ 016)


PatchTST CRPS **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 063(0 _._ 003) **0** _._ ( **0** _._ ) 0 _._ 013(0 _._ 003) **0** _._ ( **0** _._ )
MAEC 0 _._ 431(0 _._ 013) 0 _._ 406(0 _._ 076) 0 _._ 375(0 _._ 017) 0 _._ 435(0 _._ 013) 0 _._ 475(0 _._ 037) **0** _._ ( **0** _._ )

PatchTST CRPS 0 _._ 200(0 _._ 043) 0 _._ 073(0 _._ 001) **0** _._ ( **0** _._ ) 0 _._ 134(0 _._ 003) **0** _._ ( **0** _._ ) 0 _._ 056(0 _._ 024)
+Ours MAEC **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 416(0 _._ 027)


TimeMixer CRPS 0 _._ 365(0 _._ 005) 0 _._ 095(0 _._ 004) 0 _._ 273(0 _._ 006) 0 _._ 384(0 _._ 001) 0 _._ 027(0 _._ 008) **0** _._ ( **0** _._ )
MAEC 0 _._ 415(0 _._ 006) **0** _._ ( **0** _._ ) 0 _._ 427(0 _._ 001) 0 _._ 411(0 _._ 024) 0 _._ 500(0 _._ 000) 0 _._ 441(0 _._ 041)


TimeMixer CRPS **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 144(0 _._ 018)
+Ours MAEC **0** _._ ( **0** _._ ) 0 _._ 429(0 _._ 006) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )


(a) M4-Hourly (b) ETTh1 (c) Traffic


Figure 2: Histograms of the normalized fitted residuals _**η**_ - _t_ across various datasets. The red lines here
represent the Gaussian density curves based on the mean and standard deviation of _**η**_ - _t_ .


forecasting, Section B.3 of the Appendix evaluates the mean forecasting performance of various
algorithms with and without adding DualRes. All experimental results are based on five repetitions,
and we demonstrate the 95% confidence intervals apart from the average metrics.


**Results of univariate probabilistic forecasting.** The performance of DualRes is evaluated through
ablation studies in Table 1, where the baseline models are _DeepAR Salinas et al. (2020), DLinear Zeng_
_et al. (2023), PatchTST Nie et al. (2023), and TimeMixer Wang et al. (2024a)._ DLinear, PatchTST,
and TimeMixer were originally developed for mean forecasting, and their distributional indices are
obtained through fitting a t-distribution to the predictive values, which is the default operation in
probabilistic forecasting frameworks such as Alexandrov et al. (2020).


As demonstrated in Table 1, incorporating information on conditional volatility and the distribution of
normalized residuals leads to substantial improvements in both CRPS and MAEC across forecasting
algorithms—for example, the average CRPS of TimeMixer on the Exchange dataset decreases from
0.027 to 0.014 after applying DualRes. In addition, DualRes enhances the stability of forecasting
algorithms, as reflected in achieving narrower confidence intervals.


the CRPS and MAEC of various forecasting algorithms have significant decreases after incorporating
information of conditional volatility and the distribution of normalized residuals in forecasting—for
example, the average CRPS of TimeMixer when applied to Exchange data decreases from 0.027
to 0.014. Furthermore, DualRes increases the stability of the prediction algorithms in the sense of
reaching narrow confidence intervals.


We attribute the performance improvement to DualRes’s ability to capture information about both
heterogeneity and the normalized residuals distribution. As shown in Figure 3, the widths of the
prediction intervals, which are controlled by conditional volatility, vary substantially across different


7


(a) ETTh1 (b) ETTh2 (c) Electricity


(d) Traffic (e) Exchange (f) M4-Hourly


Figure 3: Prediction intervals generated by predictive algorithms incorporating DualRes. Blue lines,
red lines, and red shadow areas respectively represent the true values, the predictive means, and the
90% prediction intervals.


prediction steps. By explicitly accounting for the volatility, DualRes enhances the performance of
forecasting algorithms.


In addition to volatility, Figure 2 shows that the distribution of normalized fitted residuals rarely
follows a parametric family, such as the normal or _t_ -distribution, in real-world datasets. In practice,
these distributions may exhibit multimodality or heavy tails. DualRes avoids the need to impose
a parametric assumption—such as those in Zheng et al. (2025)—by introducing a resampling step
(Line 3 of Algorithm 2). This design also contributes to its performance gains.


5.2 MULTIVARIATE PROBABILISTIC FORECASTING


**Dataset and experimental settings.** We conduct experiments on three real-world datasets: _ETTh1,_
_ETTh2, Electricity,_ with a detailed introduction in Section B.1 of the Appendix.


Compared to univariate time series forecasting, multivariate time series data can exhibit spatial
dependence, making probabilistic forecasting algorithms essential for capturing spatial dependence.
Accordingly, in addition to CRPS and MAEC, we also evaluate the performance of probabilistic
forecasting algorithms using the energy score (ES) Chung et al. (2024), with further details provided
in Section B.2 of the Appendix.


**Results of multivariate probabilistic forecasting.** The performance of DualRes is evaluated through
ablation studies in Table 2, using baseline models _VEC-LSTM_ Salinas et al. (2019) and _TMDM_
Li et al. (2024). VEC-LSTM, also known as the DeepVAR model, is an RNN-based time series
model with a Gaussian copula process output. TMDM is a Transformer-based diffusion model. Both
algorithms were originally developed for probabilistic forecasting of multivariate time series.


According to Table 2, DualRes achieves improvements across all metrics for VEC-LSTM and for
the majority of metrics in TMDM. For example, on the _Electricity_ dataset, the CRPS of TMDM
decreases from 0.655 to 0.292 after incorporating DualRes. Apart from accounting for conditional


8


Table 2: Numerical experiment results on multivariate time series datasets. The interpretation of the
values and the use of boldface are the same as in Table 1.


Dataset ETTh1 ETTh2 Electricity


Metrics CRPS MAEC ES CRPS MAEC ES CRPS MAEC ES


VEC-LSTM 0.184(0.003) 0.310(0.015) 3.873(0.157) 0.095(0.002) 0.243(0.014) 6.423(0.196) 0.441(0.014) 0.385(0.072) 48684(3323)
+Ours **0.182(0.005)** **0.294(0.001)** **3.503(0.085)** **0.087(0.001)** **0.241(0.016)** **6.067(0.190)** **0.301(0.013)** **0.251(0.009)** **41398(3744)**


TMDM 0.456(0.023) **0.268(0.052)** 13.344(0.163) 0.092(0.008) 0.318(0.123) **6.933(0.393)** 0.655(0.275) 0.458(0.082) 87761(6179)
+Ours **0.397(0.040)** 0.458(0.082) **11.341(0.372)** **0.092(0.004)** **0.306(0.023)** 7.326(0.498) **0.292(0.018)** **0.227(0.009)** **37322(2438)**


heteroskedasticity and residual distributional information, the improvement in the energy score highlights DualRess ability to capture spatial dependence in multivariate time series. This effectiveness
stems from resampling entire normalized residual vectors _**η**_ - _t,_ rather than their individual components.


6 DISCUSSION


Focusing on probabilistic time series forecasting, this manuscript proposes the DualRes framework,
which extracts conditional volatility information from fitted residuals and models the distribution
of normalized residuals through resampling. These operations make DualRes robust to conditional
heteroskedasticity and free from restrictive parametric assumptions, such as Gaussianity. We further
provide theoretical guarantees for the validity of the proposed training and inference procedures.


In addition, as DualRes requires only conditional mean forecasts, it offers substantial flexibility in
the choice of models for both the conditional mean and volatility. As demonstrated in the numerical
experiments, even models originally designed for mean forecasting can be adapted for probabilistic
forecasting, leading to significant performance gains.


Our work highlights the importance of incorporating the distribution of normalized residuals—beyond
conditional mean and volatility—in probabilistic forecasting. Since residuals in real-world time series
often deviate from parametric distributions, introducing a resampling step enables greater flexibility
when addressing the underlying randomness in the data.


**Limitations and Future Work.** One main limitation of our work lies in the computational complexity
of the algorithm. Concerning this, one potential future direction of this work involves leveraging
advanced subsampling techniques, like those in McElroy & Politis (2024), to decrease computational
complexity.


Another limitation is that the validity of Theorem 1 depends on the conditional mean and volatility
models accurately reflecting the true conditional mean and volatility functions. As a result, if future
observations have a distributional shift, the proposed method may no longer be reliable.


REFERENCES


Alexander Alexandrov, Konstantinos Benidis, Michael Bohlke-Schneider, Valentin Flunkert, Jan
Gasthaus, Tim Januschowski, Danielle C. Maddix, Syama Rangapuram, David Salinas, Jasper
Schulz, Lorenzo Stella, Ali Caner Türkmen, and Yuyang Wang. Gluonts: Probabilistic and neural
time series modeling in python. _Journal of Machine Learning Research_, 21(116):1–6, 2020. URL
[http://jmlr.org/papers/v21/19-820.html.](http://jmlr.org/papers/v21/19-820.html)


Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen,
Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper
Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon
Wilson, Michael Bohlke-Schneider, and Bernie Wang. Chronos: Learning the language of time
series. _Transactions_ _on_ _Machine_ _Learning_ _Research_, 2024. ISSN 2835-8856. URL [https:](https://openreview.net/forum?id=gerNCVqqtR)
[//openreview.net/forum?id=gerNCVqqtR.](https://openreview.net/forum?id=gerNCVqqtR) Expert Certification.


Morgane Austern and Vasilis Syrgkanis. Asymptotics of the bootstrap via stability
with applications to inference with model selection. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), _Advances_ _in_ _Neural_ _In-_
_formation_ _Processing_ _Systems_, volume 34, pp. 10705–10717. Curran Associates, Inc.,


9


2021. URL [https://proceedings.neurips.cc/paper_files/paper/2021/](https://proceedings.neurips.cc/paper_files/paper/2021/file/58b7483ba899e0ce4d97ac5eecf6fa99-Paper.pdf)
[file/58b7483ba899e0ce4d97ac5eecf6fa99-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2021/file/58b7483ba899e0ce4d97ac5eecf6fa99-Paper.pdf)


Shane Bergsma, Tim Zeyl, and Lei Guo. Sutranets: Sub-series autoregressive networks for long-sequence, probabilistic forecasting. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), _Advances_ _in_ _Neural_ _Informa-_
_tion_ _Processing_ _Systems_, volume 36, pp. 30518–30533. Curran Associates, Inc.,
2023. URL [https://proceedings.neurips.cc/paper_files/paper/2023/](https://proceedings.neurips.cc/paper_files/paper/2023/file/6171c9e600432a42688ad61a525951bf-Paper-Conference.pdf)
[file/6171c9e600432a42688ad61a525951bf-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2023/file/6171c9e600432a42688ad61a525951bf-Paper-Conference.pdf)


Yifan Chen, Mark Goldstein, Mengjian Hua, Michael Samuel Albergo, Nicholas Matthew Boffi, and
Eric Vanden-Eijnden. Probabilistic forecasting with stochastic interpolants and föllmer processes.
In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan
Scarlett, and Felix Berkenkamp (eds.), _Proceedings_ _of_ _the_ _41st_ _International_ _Conference_ _on_
_Machine Learning_, volume 235 of _Proceedings of Machine Learning Research_, pp. 6728–6756.
PMLR, 21–27 Jul 2024a. URL [https://proceedings.mlr.press/v235/chen24n.](https://proceedings.mlr.press/v235/chen24n.html)
[html.](https://proceedings.mlr.press/v235/chen24n.html)


Yu Chen, Marin Biloš, Sarthak Mittal, Wei Deng, Kashif Rasul, and Anderson Schneider. Recurrent
interpolants for probabilistic time series prediction. _arXiv preprint arXiv:2409.11684_, 2024b.


Youngseog Chung, Ian Char, and Jeff Schneider. Sampling-based multi-dimensional recalibration. In
_Forty-first International Conference on Machine Learning_, 2024. [URL https://openreview.](https://openreview.net/forum?id=iJWeK2snMH)
[net/forum?id=iJWeK2snMH.](https://openreview.net/forum?id=iJWeK2snMH)


Kacper Chwialkowski, Dino Sejdinovic, and Arthur Gretton. A wild bootstrap for degenerate kernel tests. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 27. Curran Associates, Inc., 2014. [URL https://proceedings.neurips.cc/paper_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2014/file/4e382cb49370f64415df2672b19fb1f2-Paper.pdf)
[2014/file/4e382cb49370f64415df2672b19fb1f2-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2014/file/4e382cb49370f64415df2672b19fb1f2-Paper.pdf)


B. Efron. Bootstrap Methods: Another Look at the Jackknife. _The_ _Annals_ _of_ _Statistics_, 7(1):
1  - 26, 1979. doi: 10.1214/aos/1176344552. URL [https://doi.org/10.1214/aos/](https://doi.org/10.1214/aos/1176344552)
[1176344552.](https://doi.org/10.1214/aos/1176344552)


Shibo Feng, Chunyan Miao, Ke Xu, Jiaxiang Wu, Pengcheng Wu, Yang Zhang, and Peilin Zhao.
Multi-scale attention flow for probabilistic time series forecasting. _IEEE Trans. on Knowl. and_
_Data Eng._, 36(5):20562068, May 2024. ISSN 1041-4347. doi: 10.1109/TKDE.2023.3319672.
[URL https://doi.org/10.1109/TKDE.2023.3319672.](https://doi.org/10.1109/TKDE.2023.3319672)


Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. _Journal of the American Statistical Association_, 102(477):359–378, 2007. doi: 10.1198/
016214506000001437. [URL https://doi.org/10.1198/016214506000001437.](https://doi.org/10.1198/016214506000001437)


Hilaf Hasson, Bernie Wang, Tim Januschowski, and Jan Gasthaus. Probabilistic forecasting: A levelset approach. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan
(eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 34, pp. 6404–6416. Curran Associates, Inc., 2021. [URL https://proceedings.neurips.cc/paper_files/](https://proceedings.neurips.cc/paper_files/paper/2021/file/32b127307a606effdcc8e51f60a45922-Paper.pdf)
[paper/2021/file/32b127307a606effdcc8e51f60a45922-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2021/file/32b127307a606effdcc8e51f60a45922-Paper.pdf)


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In
H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), _Advances_ _in_ _Neu-_
_ral_ _Information_ _Processing_ _Systems_, volume 33, pp. 6840–6851. Curran Associates, Inc.,
2020. URL [https://proceedings.neurips.cc/paper_files/paper/2020/](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)


Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao
Wang, and Yuyang (Bernie) Wang. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. In A. Oh, T. Naumann,
A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), _Advances_ _in_ _Neural_ _In-_
_formation_ _Processing_ _Systems_, volume 36, pp. 28341–28364. Curran Associates, Inc.,
2023. URL [https://proceedings.neurips.cc/paper_files/paper/2023/](https://proceedings.neurips.cc/paper_files/paper/2023/file/5a1a10c2c2c9b9af1514687bc24b8f3d-Paper-Conference.pdf)
[file/5a1a10c2c2c9b9af1514687bc24b8f3d-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2023/file/5a1a10c2c2c9b9af1514687bc24b8f3d-Paper-Conference.pdf)


10


Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, and Stephan Günnemann. Flow
matching with gaussian process priors for probabilistic time series forecasting. In _The Thirteenth_
_International Conference on Learning Representations_, 2025. [URL https://openreview.](https://openreview.net/forum?id=uxVBbSlKQ4)
[net/forum?id=uxVBbSlKQ4.](https://openreview.net/forum?id=uxVBbSlKQ4)


Vincent Le Guen and Nicolas Thome. Probabilistic time series forecasting with shape and temporal diversity. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.),
_Advances in Neural Information Processing Systems_, volume 33, pp. 4427–4440. Curran Associates, Inc., 2020. [URL https://proceedings.neurips.cc/paper_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2020/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf)
[2020/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2020/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf)


Longyuan Li, Junchi Yan, Xiaokang Yang, and Yaohui Jin. Learning interpretable deep state
space model for probabilistic time series forecasting. In _Proceedings of the 28th International_
_Joint Conference on Artificial Intelligence_, IJCAI’19, pp. 29012908. AAAI Press, 2019. ISBN
9780999241141.


Yan Li, Xinjiang Lu, Yaqing Wang, and Dejing Dou. Generative time series forecasting with diffusion, denoise, and disentanglement. In S. Koyejo, S. Mohamed,
A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), _Advances_ _in_ _Neural_ _Infor-_
_mation_ _Processing_ _Systems_, volume 35, pp. 23009–23022. Curran Associates, Inc.,
2022. URL [https://proceedings.neurips.cc/paper_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf)
[file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2022/file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf)


Yuxin Li, Wenchao Chen, Xinyue Hu, Bo Chen, baolin sun, and Mingyuan Zhou. Transformermodulated diffusion models for probabilistic multivariate time series forecasting. In _The Twelfth_
_International Conference on Learning Representations_, 2024. [URL https://openreview.](https://openreview.net/forum?id=qae04YACHs)
[net/forum?id=qae04YACHs.](https://openreview.net/forum?id=qae04YACHs)


Shengsheng Lin, Weiwei Lin, Xinyi HU, Wentai Wu, Ruichao Mo, and Haocheng Zhong. Cyclenet:
Enhancing time series forecasting through modeling periodic patterns. In _The Thirty-eighth Annual_
_Conference on Neural Information Processing Systems_, 2024. [URL https://openreview.](https://openreview.net/forum?id=clBiQUgj4w)
[net/forum?id=clBiQUgj4w.](https://openreview.net/forum?id=clBiQUgj4w)


Shiqing Ling and Michael McAleer. Asymptotic theory for a vector arma-garch model. _Econometric_
_Theory_, 19(2):280310, 2003. doi: 10.1017/S0266466603192092.


Rui Luo, Weinan Zhang, Xiaojun Xu, and Jun Wang. A neural stochastic volatility model. _Proceedings_
_of the AAAI Conference on Artificial Intelligence_, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.12124.
[URL https://ojs.aaai.org/index.php/AAAI/article/view/12124.](https://ojs.aaai.org/index.php/AAAI/article/view/12124)


Tucker McElroy and Dimitris N Politis. Skip sampling: subsampling in the frequency domain.
_Biometrika_, 111(4):1241–1256, 08 2024. ISSN 1464-3510. doi: 10.1093/biomet/asae039. URL
[https://doi.org/10.1093/biomet/asae039.](https://doi.org/10.1093/biomet/asae039)


Nam Nguyen and Brian Quanz. Temporal latent auto-encoder: A method for probabilistic multivariate
time series forecasting. _Proceedings of the AAAI Conference on Artificial Intelligence_, 35(10):
9117–9125, May 2021. doi: 10.1609/aaai.v35i10.17101. [URL https://ojs.aaai.org/](https://ojs.aaai.org/index.php/AAAI/article/view/17101)
[index.php/AAAI/article/view/17101.](https://ojs.aaai.org/index.php/AAAI/article/view/17101)


Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth
64 words: Long-term forecasting with transformers. In _The_ _Eleventh_ _International_ _Confer-_
_ence on Learning Representations_, 2023. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=Jbdc0vTOcol)
[Jbdc0vTOcol.](https://openreview.net/forum?id=Jbdc0vTOcol)


Li Pan and Dimitris N. Politis. Bootstrap prediction intervals for linear, nonlinear and nonparametric
autoregressions. _Journal of Statistical Planning and Inference_, 177:1–27, 2016. ISSN 0378-3758.
doi: https://doi.org/10.1016/j.jspi.2014.10.003. [URL https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S037837581400175X)
[science/article/pii/S037837581400175X.](https://www.sciencedirect.com/science/article/pii/S037837581400175X)


Dimitris N. Politis, Joseph P. Romano, and Michael Wolf. _Subsampling_ . Springer Series in Statistics.
Springer-Verlag, New York, 1999. ISBN 0-387-98854-8. doi: 10.1007/978-1-4612-1554-7. URL
[https://doi.org/10.1007/978-1-4612-1554-7.](https://doi.org/10.1007/978-1-4612-1554-7)


11


Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
and Tim Januschowski. Deep state space models for time series forecasting. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.),
_Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 31. Curran Associates, Inc.,
2018. URL [https://proceedings.neurips.cc/paper_files/paper/2018/](https://proceedings.neurips.cc/paper_files/paper/2018/file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf)
[file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2018/file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf)


Syama Sundar Rangapuram, Lucien D Werner, Konstantinos Benidis, Pedro Mercado, Jan Gasthaus,
and Tim Januschowski. End-to-end learning of coherent probabilistic forecasts for hierarchical
time series. In Marina Meila and Tong Zhang (eds.), _Proceedings_ _of_ _the_ _38th_ _International_
_Conference on Machine Learning_, volume 139 of _Proceedings of Machine Learning Research_, pp.
8832–8843. PMLR, 18–24 Jul 2021. [URL https://proceedings.mlr.press/v139/](https://proceedings.mlr.press/v139/rangapuram21a.html)
[rangapuram21a.html.](https://proceedings.mlr.press/v139/rangapuram21a.html)


Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising
diffusion models for multivariate probabilistic time series forecasting. In Marina Meila and Tong
Zhang (eds.), _Proceedings of the 38th International Conference on Machine Learning_, volume 139
of _Proceedings of Machine Learning Research_, pp. 8857–8868. PMLR, 18–24 Jul 2021a. URL
[https://proceedings.mlr.press/v139/rasul21a.html.](https://proceedings.mlr.press/v139/rasul21a.html)


Kashif Rasul, Abdul-Saboor Sheikh, Ingmar Schuster, Urs M Bergmann, and Roland Vollgraf.
Multivariate probabilistic time series forecasting via conditioned normalizing flows. In _Interna-_
_tional Conference on Learning Representations_, 2021b. [URL https://openreview.net/](https://openreview.net/forum?id=WiGQBFuVRv)
[forum?id=WiGQBFuVRv.](https://openreview.net/forum?id=WiGQBFuVRv)


Raanan Y. Rohekar, Yaniv Gurwicz, Shami Nisimov, Guy Koren, and Gal Novik. Bayesian structure
learning by recursive bootstrap. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. CesaBianchi, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_, volume 31.
Curran Associates, Inc., 2018. URL [https://proceedings.neurips.cc/paper_](https://proceedings.neurips.cc/paper_files/paper/2018/file/11e2ad6bf99300cd3808bb105b55d4b8-Paper.pdf)
[files/paper/2018/file/11e2ad6bf99300cd3808bb105b55d4b8-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2018/file/11e2ad6bf99300cd3808bb105b55d4b8-Paper.pdf)


David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico, and Jan Gasthaus. _High-_
_dimensional multivariate forecasting with low-rank Gaussian copula processes_ . Curran Associates
Inc., Red Hook, NY, USA, 2019.


David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. _International Journal of Forecasting_, 36(3):1181–1191,
2020. ISSN 0169-2070. doi: https://doi.org/10.1016/j.ijforecast.2019.07.001. URL [https:](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
[//www.sciencedirect.com/science/article/pii/S0169207019301888.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)


Olimjon Shukurovich Sharipov. _Glivenko-Cantelli Theorems_, pp. 612–614. Springer Berlin Heidelberg, Berlin, Heidelberg, 2011. ISBN 978-3-642-04898-2. doi: 10.1007/978-3-642-04898-2_280.
[URL https://doi.org/10.1007/978-3-642-04898-2_280.](https://doi.org/10.1007/978-3-642-04898-2_280)


Minsuk Shin, Hyungjoo Cho, Hyun-seok Min, and Sungbin Lim. Neural bootstrapper. In
M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), _Ad-_
_vances in Neural Information Processing Systems_, volume 34, pp. 16596–16609. Curran Associates, Inc., 2021. [URL https://proceedings.neurips.cc/paper_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2021/file/8abfe8ac9ec214d68541fcb888c0b4c3-Paper.pdf)
[2021/file/8abfe8ac9ec214d68541fcb888c0b4c3-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2021/file/8abfe8ac9ec214d68541fcb888c0b4c3-Paper.pdf)


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint_
_arXiv:2011.13456_, 2020.


Robert A. Stine. Bootstrap prediction intervals for regression. _Journal of the American Statistical_
_Association_, 80(392):1026–1031, 1985. ISSN 01621459. [URL http://www.jstor.org/](http://www.jstor.org/stable/2288570)
[stable/2288570.](http://www.jstor.org/stable/2288570)


Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: conditional score-based
diffusion models for probabilistic time series imputation. In _Proceedings of the 35th International_
_Conference on Neural Information Processing Systems_, NIPS ’21, Red Hook, NY, USA, 2021.
Curran Associates Inc. ISBN 9781713845393.


12


Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y Zhang,
and JUN ZHOU. Timemixer: Decomposable multiscale mixing for time series forecasting. In
_International Conference on Learning Representations (ICLR)_, 2024a.


Yaoming Wang, Jin Li, Wenrui Dai, Bowen Shi, Xiaopeng Zhang, Chenglin Li, and Hongkai Xiong.
Bootstrap AutoEncoders with contrastive paradigm for self-supervised gaze estimation. In Ruslan
Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and
Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_,
volume 235 of _Proceedings of Machine Learning Research_, pp. 50794–50806. PMLR, 21–27 Jul
2024b. [URL https://proceedings.mlr.press/v235/wang24ah.html.](https://proceedings.mlr.press/v235/wang24ah.html)


Martha White and Adam White. Interval estimation for reinforcement-learning algorithms in
continuous-state domains. In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 23. Curran Associates, Inc., 2010. [URL https://proceedings.neurips.cc/paper_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2010/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf)
[2010/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2010/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf)


C. F. J. Wu. Jackknife, Bootstrap and Other Resampling Methods in Regression Analysis. _The_
_Annals_ _of_ _Statistics_, 14(4):1261  - 1295, 1986. doi: 10.1214/aos/1176350142. URL [https:](https://doi.org/10.1214/aos/1176350142)
[//doi.org/10.1214/aos/1176350142.](https://doi.org/10.1214/aos/1176350142)


Kejin Wu and Dimitris N. Politis. Bootstrap prediction inference of nonlinear autoregressive models.
_Journal of Time Series Analysis_, 45(5):800–822, 2024. doi: https://doi.org/10.1111/jtsa.12739.
[URL https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.12739.](https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.12739)


Kejin Wu and Dimitris N. Politis. Scalable subsampling inference for deep neural networks. _ACM /_
_IMS J. Data Sci._, January 2025. doi: 10.1145/3711709. [URL https://doi.org/10.1145/](https://doi.org/10.1145/3711709)
[3711709.](https://doi.org/10.1145/3711709) Just Accepted.


Mengyu Xu, Danna Zhang, and Wei Biao Wu. Pearsons chi-squared statistics: approximation theory
and beyond. _Biometrika_, 106(3):716–723, 04 2019. ISSN 0006-3444. doi: 10.1093/biomet/asz020.
[URL https://doi.org/10.1093/biomet/asz020.](https://doi.org/10.1093/biomet/asz020)


Weiwei Ye, Zhuopeng Xu, and Ning Gui. Non-stationary diffusion for probabilistic time series
forecasting, 2025. [URL https://arxiv.org/abs/2505.04278.](https://arxiv.org/abs/2505.04278)


Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo
Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for
large language models. In _The Twelfth International Conference on Learning Representations_,
2024. [URL https://openreview.net/forum?id=N8N0hgNDRt.](https://openreview.net/forum?id=N8N0hgNDRt)


Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series
forecasting? _Proceedings of the AAAI Conference on Artificial Intelligence_, 37(9):11121–11128,
Jun. 2023. doi: 10.1609/aaai.v37i9.26317. [URL https://ojs.aaai.org/index.php/](https://ojs.aaai.org/index.php/AAAI/article/view/26317)
[AAAI/article/view/26317.](https://ojs.aaai.org/index.php/AAAI/article/view/26317)


Yaoli Zhang, Ye Tian, and Yunyi Zhang. Leveraging temporal dependency in probabilistic electric
load forecasting. _Applied Soft Computing_, 169:112611, 2025. ISSN 1568-4946. doi: https://doi.
org/10.1016/j.asoc.2024.112611. [URL https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/S1568494624013851)
[article/pii/S1568494624013851.](https://www.sciencedirect.com/science/article/pii/S1568494624013851)


Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency
for multivariate time series forecasting. In _The Eleventh International Conference on Learning_
_Representations_, 2023. [URL https://openreview.net/forum?id=vSVLM2j9eie.](https://openreview.net/forum?id=vSVLM2j9eie)


Yunyi Zhang, Efstathios Paparoditis, and Dimitris N. Politis. Simultaneous statistical inference for
second order parameters of time series under weak conditions. _The Annals of Statistics_, 52(5):2375 –
2399, 2024. doi: 10.1214/24-AOS2439. [URL https://doi.org/10.1214/24-AOS2439.](https://doi.org/10.1214/24-AOS2439)


Ronghua Zheng, Hanru Bai, and Weiyang Ding. KooNPro: A variance-aware koopman probabilistic
model enhanced by neural process for time series forecasting. In _The Thirteenth International_
_Conference on Learning Representations_, 2025. [URL https://openreview.net/forum?](https://openreview.net/forum?id=5oSUgTzs8Y)
[id=5oSUgTzs8Y.](https://openreview.net/forum?id=5oSUgTzs8Y)


13


A PROOF OF THEOREM 1


To validate Theorem 1, we propose the following technical assumptions.


**Assumptions:**


1. _**η**_ _t, t_ = 1 _,_ 2 _, · · ·_ _,_ are independent and identically distributed with continuous cumulative distribution function _P_ ( _·_ ) : R _[d]_ _→_ R. Suppose **E** [ _**η**_ 1] = 0 and Var( _**η**_ 1 _,i_ ) _≤_ _C_ for a constant _C_ and any
_i_ = 1 _, · · ·_ _, d_ .


2. For a vector _**x**_ _∈_ **R** _[d]_ _,_ define _||_ _**x**_ _||_ as its _L_ [2] norm. We suppose the conditional mean and volatility
function estimator satisfy


sup [0] and sup [0] _[,]_
**Y** _∈_ **R** _[d][×][q][ ||][F]_ [ �][(] **[Y]** [)] _[ −]_ _[F]_ [(] **[Y]** [)] _[|| →][p]_ **Y** _∈_ **R** _[d][×][s][ |][G]_ [ �] _[i]_ [(] **[Y]** [)] _[ −]_ _[G][i]_ [(] **[Y]** [)] _[| →][p]_


where _i_ = 1 _,_ 2 _, · · ·_ _, d,_ and _→p_ denotes convergence in probability.


3. Suppose _Gi_ ( _·_ ) is continuous differentiable with bounded gradient, i.e.,


sup
**Y** _∈_ **R** _[d][×][s][ ||∇]_ **[Y]** _[G][i]_ [(] **[Y]** [)] _[||][ <][ ∞]_


for _i_ = 1 _, · · ·_ _, d._ Furthermore, suppose there exists a constant _c >_ 0 such that


inf
**Y** _∈_ **R** _[d][×][s][ |][G][i]_ [(] **[Y]** [)] _[|][ > c]_


for _i_ = 1 _, · · ·_ _, d._


With those assumptions, we demonstrate that Theorem 1 holds true.


_Proof of Theorem 1._ For any vector **y** = ( **y** 1 _, · · ·_ _,_ **y** _d_ ) _[⊤]_ _∈_ **R** _[d]_ _,_ define


1
_F_ �( _**y**_ ) = _T_ _−_ _q −_ _s_


1
_P_ �( _**y**_ ) = _T_ _−_ _q −_ _s_


_T_

 - **1** _**η**_ _t≤_ _**y**_ _._


_t_ = _s_ + _q_ +1


From Glivenko-Cantelli Theorem, like Theorem 4 of Sharipov (2011), we have


sup [0] _[.]_
_**y**_ _∈_ **R** _[d][ |][G]_ [ �][(] _**[y]**_ [)] _[ −]_ _[G]_ [(] _**[y]**_ [)] _[| →][p]_


On the other hand, define the functions


_g_ 0( _u_ ) = (1 _−_ min(1 _,_ max( _u,_ 0)) [4] ) [4] and _gψ,t_ ( _x_ ) = _g_ 0( _ψ_ ( _x −_ _t_ )) _,_


as demonstrated in Xu et al. (2019), which satisfy the following property: _g_ 0( _·_ ) is third-order
continuous differentiable, _g_ 0( _u_ ) = 1 if _u ≤_ 0, _g_ 0( _u_ ) = 0 if _u ≥_ 1, and


_g∗_ = sup _{|g_ 0 _[′]_ [(] _[u]_ [)] _[|]_ [ +] _[ |][g]_ 0 _[′′]_ [(] _[u]_ [)] _[|]_ [ +] _[ |][g]_ 0 _[′′′]_ [(] _[u]_ [)] _[|}][ <][ ∞][,]_ **[1]** _[x][≤][t]_ _[≤]_ _[g][ψ,t]_ [(] _[x]_ [)] _[ ≤]_ **[1]** _x≤t_ + _ψ_ _[−]_ [1] _[,]_ [sup] _|gψ,t_ _[′]_ [(] _[x]_ [)] _[| ≤]_ _[g][∗][ψ.]_
_u∈_ **R** _x,t∈_ **R**


Define


**∆** _t_ = _**η**_   - _t −_ _**η**_ _t_

��            = _G_ [�] _[−]_ [1][ �] _**ζ**_     - _t−s, · · ·_     - _**ζ**_ _t−_ 1 _F_ ( **x** _t−q, · · ·_ _,_ **x** _t−_ 1) _−_ _F_ [�] ( **x** _t−q, · · ·_ _,_ **x** _t−_ 1)


��            - ��
+ _G_ [�] _[−]_ [1][ �] _**ζ**_     - _t−s, · · ·_     - _**ζ**_ _t−_ 1 _G_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _G_ [�] _**ζ**_     - _t−s, · · ·_     - _**ζ**_ _t−_ 1 _**η**_ _t._


Notice that


_T_

 

_t_ = _s_ + _q_ +1


_d_

- _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ + **∆** _t,i_ ) _._


_i_ =1


_T_

 - **1** _**η**_ _t_ + **∆** _t≤_ _**y**_ _≤_ _T_ _−_ 1 _q −_ _s_

_t_ = _s_ + _q_ +1


14


From Taylor expansion,


With probability tending to 1,


inf inf sup
**Y** _∈_ **R** _[d][×][s]_ _[G]_ [�] _[i]_ [(] **[Y]** [)] _[ ≥]_ **Y** _∈_ **R** _[d][×][s][ G][i]_ [(] **[Y]** [)] _[ −]_ **Y** _∈_ **R** _[d][×][s][ |][G]_ [ �] _[i]_ [(] **[Y]** [)] _[ −]_ _[G][i]_ [(] **[Y]** [)] _[|][ > c/]_ [2] _[.]_


If that happens for _i_ = 1 _, · · ·_ _, d,_ we have


��           _||G_ [�] _[−]_ [1][ �] _**ζ**_       - _t−s, · · ·_       - _**ζ**_ _t−_ 1 _F_ ( **x** _t−q, · · ·_ _,_ **x** _t−_ 1) _−_ _F_ [�] ( **x** _t−q, · · ·_ _,_ **x** _t−_ 1) _||_

(9)

_≤_ [2] sup [0] _[.]_

_c_ **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[|| →][p]_


On the other hand, for any _i_ = 1 _, · · ·_ _, d,_ the _i_ th element of
��       - ��
_G_ - _[−]_ [1][ �] _**ζ**_ - _t−s, · · ·_ - _**ζ**_ _t−_ 1 _G_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _G_ [�] _**ζ**_ - _t−s, · · ·_ - _**ζ**_ _t−_ 1 _**η**_ _t_ is


                  -                  _Gi_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _G_ [�] _i_ _**ζ**_           - _t−s, · · ·_           - _**ζ**_ _t−_ 1

              -              - _**η**_ _t,i._
_G_          - _i_ _**ζ**_          - _t−s, · · ·_          - _**ζ**_ _t−_ 1


15


_d_

- _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ ) _|_


_i_ =1


_|_


_d_

- _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ + **∆** _t,i_ ) _−_


_i_ =1


_i−_ 1


_gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ + **∆** _t,i_ )( _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ + **∆** _t,i_ ) _−_ _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ ))

_j_ =1


_≤_


_≤_


Therefore,


_d_


_d_

- _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ )


_j_ = _i_ +1


(

_i_ =1


_d_

- _|gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ + **∆** _t,i_ ) _−_ _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ ) _| ≤_ _g∗ψ_


_i_ =1


_d_ _√_

- _|_ **∆** _t,i| ≤_ _g∗ψ_


_i_ =1


_d||_ **∆** _t||._


_d_

- _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ + **∆** _t,i_ )


_i_ =1


1
_T_ _−_ _q −_ _s_


_T_

 

_t_ = _s_ + _q_ +1


_d_ _√_

- _gψ,_ _**y**_ _i_ ( _**η**_ _t,i_ ) + _Tg−∗ψq_

_i_ =1


_d_


_g∗ψ_ _d_

_T_ _−_ _q −_ _s_


1
_≤_
_T_ _−_ _q −_ _s_


1
_≤_
_T_ _−_ _q −_ _s_


_T_

 

_t_ = _s_ + _q_ +1


_T_ _√_

 - _g∗ψ_

**1** _**η**_ _t≤_ _**y**_ + _ψ−_ 1 + _T_ _−_ _q_
_t_ = _s_ + _q_ +1


_T_

 - _||_ **∆** _t||_


_t_ = _s_ + _q_ +1


_T_


_g∗ψ_ _d_

_T_ _−_ _q −_ _s_


_T_

 - _||_ **∆** _t||_


_t_ = _s_ + _q_ +1


_√_
_g∗ψ_ _d_
= _F_ ( _**y**_ + _ψ_ _[−]_ [1] **h** ) +

[�] _T_ _−_ _q −_ _s_


_T_

 - _||_ **∆** _t||,_


_t_ = _s_ + _q_ +1


where **h** = (1 _,_ 1 _, · · ·_ _,_ 1) _[⊤]_ . Similarly,


_d_


_gψ,_ _**y**_ _i−ψ−_ 1( _**η**_ _t,i_ + **∆** _t,i_ )
_i_ =1


_d_ _√_

- _g∗ψ_

_gψ,_ _**y**_ _i−ψ−_ 1( _**η**_ _t,i_ ) _−_ _T_ _−_ _q_
_i_ =1


1
_F_ �( _**y**_ ) _≥_ _T_ _−_ _q −_ _s_


1
_≥_
_T_ _−_ _q −_ _s_


_T_

 

_t_ = _s_ + _q_ +1


_T_

 

_t_ = _s_ + _q_ +1


_d_


_g∗ψ_ _d_

_T_ _−_ _q −_ _s_


_T_

 - _||_ **∆** _t||_


_t_ = _s_ + _q_ +1


_√_
_g∗ψ_ _d_
_≥_ _F_ [�] ( _**y**_ _−_ _ψ_ _[−]_ [1] **h** ) _−_

_T_ _−_ _q −_ _s_


_T_

 - _||_ **∆** _t||._


_t_ = _s_ + _q_ +1


and


                -                 _Gi_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _G_ [�] _i_ _**ζ**_         - _t−s, · · ·_         - _**ζ**_ _t−_ 1
_|_                -                - _**η**_ _t,i|_

_G_         - _i_ _**ζ**_         - _t−s, · · ·_         - _**ζ**_ _t−_ 1


_√_

_d_

+ [2]

_c_


_≤_ [2] _[|]_ _**[η]**_ _[t,i][|]_

_c_


- - _|Gi_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _Gi_ _**ζ**_ - _t−s, · · ·_ - _**ζ**_ _t−_ 1 _|_


          -          -          -          -          + _|Gi_ _**ζ**_      - _t−s, · · ·_      - _**ζ**_ _t−_ 1 _−_ _G_ [�] _i_ _**ζ**_      - _t−s, · · ·_      - _**ζ**_ _t−_ 1 _|_


From Assumption 2,


    -     -     -     _|Gi_ _**ζ**_    - _t−s, · · ·_    - _**ζ**_ _t−_ 1 _−_ _G_ [�] _i_ _**ζ**_    - _t−s, · · ·_    - _**ζ**_ _t−_ 1 _| ≤_ sup [0] _[.]_ (10)
**Y** _∈_ **R** _[d][×][s][ |][G][i]_ [ (] **[Y]** [)] _[ −]_ _[G]_ [�] _[i]_ [ (] **[Y]** [)] _[ | →][p]_


On the other hand, for any _t_ = _q_ + 1 _, · · ·_ _, T,_


_||_ _**ζ**_ [�] _t −_ _**ζ**_ _t||_ = _||F_ ( **x** _t−q, · · ·_ _,_ **x** _t−_ 1) _−_ _F_ [�] ( **x** _t−q, · · ·_ _,_ **x** _t−_ 1) _||_


_≤_ sup [0] _[.]_
**Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[|| →][p]_


Define the matrix


**Γ** = �� _**ζ**_ _t−s −_ _**ζ**_ _t−s_ _· · ·_ _**ζ**_         - _t−_ 1 _−_ _**ζ**_ _t−_ 1� _,_


from Taylor’s expansion,


_d_

_|Gi_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _Gi_ - _**ζ**_ - _t−s, · · ·_ - _**ζ**_ _t−_ 1� _|_ = _|_ 
_i_ =1


_s_
�( _∇_ **Z** _Gi_ ( **Z** )) _ij_ **Γ** _ij|_


_j_ =1


_s_

- _|∇_ **Z** _Gi_ ( **Z** )) _ij||_ **Γ** _ij|_


_j_ =1


_≤_


_d_


_i_ =1


(11)


_≤_ _Cds_ sup
**Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||][,]_


where **Z** _∈_ **R** _[d][×][s]_ is a random matrix. From eq.equation 9, eq.equation 10 and eq.equation 11, with
probability tending to 1


 - 
_Gi_ ( _**ζ**_ _t−s, · · ·_ _**ζ**_ _t−_ 1) _−_ _G_ [�] _i_ _**ζ**_  - _t−s, · · ·_  - _**ζ**_ _t−_ 1






_||_ **∆** _t|| ≤_ [2] sup

_c_ **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_ [ +]


~~�~~


_d_


_i_ =1


2







 -  - _**η**_ _t,i_
_G_ - _i_ _**ζ**_ - _t−s, · · ·_ - _**ζ**_ _t−_ 1


_√_

[2] sup [2]

_c_ **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_ [ +] _c_

_√_

[2] sup [2]

_c_ **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_ [ +] _c_


_≤_ [2]


_d_ - _c_ _i_ =1max _,···,d_ _[|]_ _**[η]**_ _[t,i][| × |][G][i]_ [ (] _**[ζ]**_ _[t][−][s][,][ · · ·]_ _**[ ζ]**_ _[t][−]_ [1][)] _[ −]_ _[G]_ [�] _[i]_ _**ζ**_ - _t−s, · · ·_ - _**ζ**_ _t−_ 1 _|_


�� sup
**Y** _∈_ **R** _[d][×][s][ |][G][i]_ [ (] **[Y]** [)] _[ −]_ _[G]_ [�] _[i]_ [ (] **[Y]** [)] _[ |]_


_≤_ [2]


_d_

_c_


- _d_

 


_|_ _**η**_ _t,i|_

_i_ =1


_|_ _**η**_ _t,i|_

_i_ =1


- _d_

 


�� _Cds_ sup _._
**Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_


16


and the result is proven according to the continuity of _P_ ( _·_ ) _,_ and by setting _ψ_ _→∞_ .


B ADDITIONAL EXPERIMENTAL RESULTS


B.1 INTRODUCTION OF DATASETS AND HYPER-PARAMETERS


Our work evaluates the performance of models on six commonly used datasets named _ETTh1, ETTh2,_
_Electricity, Traffic, Exchange, M4-Hourly_ when performing univariate probabilistic forecasting, and
on three datasets _ETTh1, ETTh2, Electricity_ when performing multivariate probabilistic forecasting.
The names and characteristics of the datasets are summarized as in Table 3. _Electricity,_ _Traffic,_
_Exchange, M4-Hourly_ are available in GluonTS Alexandrov et al. (2020). We consider the _ETTh1,_
_ETTh2, Electricity_ datasets as multiple separate univariate time series in univariate experiments, while
we consider them as single multivariate time series data in multivariate experiments.


Table 3: Overview of the datasets used in univariate time series experiments.


Dataset GluonTS Name Dimension Test Domain Freq. Median Time Steps


ETTh1 [1]      - 7 126 **R** [+] H 17396
ETTh2 [2]      - 7 126 **R** [+] H 17396
M4-Hourly [3] m4_hourly 414 414 N H 960
Electricity [4] electricity_nips 370 2590 **R** [+] H 5833
Traffic [5] traffic_nips 963 6741 (0 _,_ 1) H 4001
Exchange [6] exchange_rate_nips 8 40 **R** [+] D 6071


[1https://github.com/zhouhaoyi/ETDataset/tree/main](https://github.com/zhouhaoyi/ETDataset/tree/main)
[2https://github.com/zhouhaoyi/ETDataset/tree/main](https://github.com/zhouhaoyi/ETDataset/tree/main)
[3https://github.com/Mcompetitions/M4-methods/tree/master/Dataset](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset)
[4ttps://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014](ttps://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
[5https://zenodo.org/records/4656132](https://zenodo.org/records/4656132)
[6https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)


17


Since


_√_
_ψ_


_d_
sup
_c_ **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_


_ψ_ _d_

_T_ _−_ _q −_ _s_


_T_ _√_

 - _||_ **∆** _t|| ≤_ [2] _[ψ]_

_c_

_t_ = _s_ + _q_ +1


_T_


_T_

 - _|_ _**η**_ _t,i|_


_t_ = _s_ + _q_ +1


2 _ψd_
+
_c_ ( _T_ _−_ _q −_ _s_ )


_d_

- sup

_i_ =1 **Y** _∈_ **R** _[d][×][s][ |][G][i]_ [ (] **[Y]** [)] _[ −]_ _[G]_ [�] _[i]_ [ (] **[Y]** [)] _[ |]_


_T_

 - _|_ _**η**_ _t,i|_


_t_ = _s_ + _q_ +1


2 _Cψd_ [2] _s_
+ sup
_c_ ( _T_ _−_ _q −_ _s_ ) **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_

_√_

_d_

_≤_ [2] _[ψ]_ sup

_c_ **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_


_d_


_i_ =1


2 _ψd_
+
_c_ ( _T_ _−_ _q −_ _s_ )


- - [�] _d_
max sup  _i_ =1 _,···,d_ **Y** _∈_ **R** _[d][×][s][ |][G][i]_ [ (] **[Y]** [)] _[ −]_ _[G]_ [�] _[i]_ [ (] **[Y]** [)] _[ |]_ _i_ =1


- - [�] _d_
max sup  _i_ =1 _,···,d_ **Y** _∈_ **R** _[d][×][s][ |][G][i]_ [ (] **[Y]** [)] _[ −]_ _[G]_ [�] _[i]_ [ (] **[Y]** [)] _[ |]_


_|_ _**η**_ _t,i|_

_t_ = _s_ + _q_ +1


_T_


_T_

 - _|_ _**η**_ _t,i|,_


_t_ = _s_ + _q_ +1


2 _Cψd_ [2] _s_
+ sup
_c_ ( _T_ _−_ _q −_ _s_ ) **Y** _∈_ **R** _[d][×][q][ ||][F]_ [(] **[Y]** [)] _[ −]_ _[F]_ [�][(] **[Y]** [)] _[||]_


_d_


_i_ =1


and


_d_


_i_ =1


_|_ _**η**_ _t,i|_

_t_ = _s_ + _q_ +1


_T_


=


**E**


1
_T_ _−_ _q −_ _s_


_d_

- **E** [ _|_ _**η**_ 1 _,i|_ ] _< ∞._


_i_ =1


According to Assumption 2,


_√_
_ψ_ _d_

_T_ _−_ _q −_ _s_


_T_

 - _||_ **∆** _t|| →p_ 0 _,_


_t_ = _s_ + _q_ +1


For the experiment detail, we set the resample times 100 when computing the CRPS and MAEC
metrics. The context length and prediction length in conditional mean model follow the settings
in Kollovieh et al. (2023). In our work, for univariate time series data, we use the technique
mentioned in Remarks 1 and 2, and adopt a simple multilayer perceptron model (referred to as
“SimpleFeedForwardEstimator in the _GluonTS_ package Alexandrov et al. (2020)) to model the
logarithm of the conditional volatilities. For multivariate time series, we use the _VEC-LSTM_ model
to estimate the logarithm of conditional volatilities in the first experiment, and the _TMDM_ model
in the second one. The context length of the conditional volatility model is selected based on the
autocorrelation coefficients plot (Figure 4) below. The prediction length in the conditional volatility
model is set to 1. All other hyperparameters are set to their default values in the GluonTS package.


Table 4: Hyperparameters of the Conditional Mean and Volatility model


Conditional Mean Model Conditional Volatility Model


Dataset Context Len. Predict Len. Context Len. Predict Len.


ETTh1 336 24 24 1
ETTh2 336 24 24 1
M4-Hourly 312 48 14 1
Electricity 336 24 48 1
Traffic 336 24 48 1
Exchange 360 30 100 1


(a) ETTh1 (b) ETTh2 (c) M4-Hourly


(d) Electricity (e) Traffic (f) Exchange


Figure 4: Autocorrelation coefficients plot of the logarithm of square fitted residuals.


B.2 METRICS OF THE EXPERIMENT


**Continuous Ranked Probability Score (CRPS).** The CRPS is a commonly used metric in probabilistic forecasting, as demonstrated in Gneiting & Raftery (2007) and Kollovieh et al. (2023). It is
defined as the integral of the pinball loss over the interval [0 _,_ 1]:


          - 1
_CRPS_ ( _F_ _[−]_ [1] _, y_ ) = 2Λ _κ_ ( _F_ _[−]_ [1] ( _κ_ ) _, y_ )d _κ,_ where Λ _κ_ ( _q, y_ ) = ( _κ −_ **1** _y<q_ ) _×_ ( _y −_ _q_ ) _._

0

A forecasted quantile function _F_ _[−]_ [1] with a small CRPS indicates good alignment with the observation _y._ We approximate the quantile function by sample quantiles at nine quantile levels
_{_ 10% _,_ 20% _, · · ·_ _,_ 90% _}._ These sample quantiles are estimated from 100 forecast samples.


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


For multivariate time series, the CRPS is computed as the summation of the element-wise CRPS.


**Mean Absolute Error of Coverage (MAEC).** Suppose the prediction step is _J,_ and the prediction
intervals are with endpoints **u** _j,_ **v** _j_ _∈_ **R** _[d]_ _,_ where **u** _j,i_ _≤_ **v** _j,i_ for _i_ = 1 _, · · ·_ _, d,_ here _j_ = 1 _, · · ·_ _, J._
The coverage probability we are interested in is the frequency


_d_

- **1u** _j,i≤_ **x** _T_ + _j,i≤_ **v** _j,i,_


_i_ =1


_p_ ( _β_ ) = [1]

- _dJ_


_J_


_j_ =1


_β_ here indicates the quantile level of the prediction intervals. Specifically, for univariate time series
( _d_ = 1), the endpoints of prediction intervals are scalars, and the coverage probability becomes


_p_ ( _β_ ) = [1]

- _J_


_J_

- **1u** _j,_ 1 _≤_ **x** _T_ + _j_ _≤_ **v** _j,_ 1 _._


_j_ =1


We consider 9 quantile levels _{β_ 1 _, · · ·_ _, β_ 9 _}_ = 10% _,_ 20% _, · · ·_ _,_ 90% _,_ and the MAEC metric calculates
the mean absolute error between _p_ �( _βs_ ) and _βs,_ i.e.,


_MAEC_ =


9

- _|p_ �( _βs_ ) _−_ _βs|._

_s_ =1


A low MAEC indicates that the prediction intervals achieve the desired coverage probabilities in
general, thereby reflecting higher accuracy of prediction intervals.


**Energy Score (ES).** Introduced in Chung et al. (2024), ES is a metric to evaluate the performance of
a probabilistic forecasting method in capturing spatial dependence for multivariate data. For a future
time series data **y** _j_ _∈_ **R** _[d]_ _,_ and a predictive distribution _p_ - _j,_ we define the energy score as

_ESj_ = E **x** _∼p_         - _j_ _||_ **x** _−_ **y** _j||_ _[β]_ 2 _[−]_ [1] 2 [E] **[x]** _[,]_ **[x]** _[′][∼][p]_ [�] _[j]_ _[||]_ **[x]** _[ −]_ **[x]** _[′][||]_ 2 _[β][,]_

where **x** _,_ **x** _[′]_ are independent sampled from _p_ - _j._ We calculate the ES as the average value


_ES_ = [1]

_J_


_J_

- _ESj._


_j_ =1


Following Chung et al. (2024), we set _β_ = 1 _._ 7 _._ A smaller energy score indicates that the predictive
distribution is closer to the ground truth.


In addition to the probabilistic forecasting metrics, we evaluate the mean forecasting performance
of univariate time series through the metrics _Normalized Deviation (ND)_ and _normalized root mean_
_squared error (NRMSE),_ introduced as follows:


**Normalized Deviation (ND).** Suppose the future _J_ observations are **x** _T_ +1 _, · · ·_ _,_ **x** _T_ + _J_ with corresponding predictors � **x** _T_ + _j,_ ND is defined by


_ND_ =


- _Jj_ =1 _[|]_ **[x]** [�] _[T]_ [ +] _[j]_ _[−]_ **[x]** _[T]_ [ +] _[j][|]_

_,_

 - _J_
_j_ =1 _[|]_ **[x]** _[T]_ [ +] _[j][|]_


indicating the absolute error normalized by the total absolute scale of the prediction time series. ND
is independent of the scale of the time series, making it suitable for comparison across different
datasets.


**Normalized root mean squared error (NRMSE).** With the notations in ND, the NRMSE is defined
by


�( **x** - _T_ + _j_ _−_ **x** _T_ + _j_ ) [2] and _|_ **x** _|_ = _J_ [1]

_j_ =1


_J_


_RMSE_

_,_ where _RMSE_ =
_|_ **x** _|_


~~�~~


- [1]
_J_


_J_


_J_

- _|_ **x** _T_ + _j|._


_j_ =1


Similar to ND, NRMSE is also independent of the scale of time series.


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


Table 5: Mean forecasting performance. The interpretation of the values and the use of boldface are
the same as in Table 1.


Models Metrics ETTh1 ETTh2 Electricity Traffic Exchange M4-Hourly


DeepAR ND 0 _._ 225(0 _._ 045) **0** _._ ( **0** _._ ) 0 _._ 104(0 _._ 001) **0** _._ ( **0** _._ ) 0 _._ 019(0 _._ 002) 0 _._ 109(0 _._ 113)
NRMSE 0 _._ 417(0 _._ 063) 0 _._ 123(0 _._ 015) 0 _._ 760(0 _._ 010) **0** _._ ( **0** _._ ) 0 _._ 029(0 _._ 002) 0 _._ 653(0 _._ 515)

DeepAR ND **0** _._ ( **0** _._ ) 0 _._ 114(0 _._ 017) **0** _._ ( **0** _._ ) 0 _._ 154(0 _._ 010) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )
+Ours NRMSE **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 429(0 _._ 037) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )


DLinear ND **0** _._ ( **0** _._ ) 0 _._ 086(0 _._ 011) 0 _._ 075(0 _._ 009) 0 _._ 161(0 _._ 002) 0 _._ 024(0 _._ 010) 0 _._ 057(0 _._ 006)
NRMSE **0** _._ ( **0** _._ ) 0 _._ 126(0 _._ 012) 0 _._ 593(0 _._ 063) **0** _._ ( **0** _._ ) 0 _._ 044(0 _._ 026) 0 _._ 323(0 _._ 050)


DLinear ND 0 _._ 243(0 _._ 011) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )
+Ours NRMSE 0 _._ 452(0 _._ 016) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 418(0 _._ 001) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )


PatchTST ND **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 017(0 _._ 005) **0** _._ ( **0** _._ )
NRMSE **0** _._ ( **0** _._ ) 0 _._ 122(0 _._ 021) **0** _._ ( **0** _._ ) 0 _._ 441(0 _._ 003) 0 _._ 024(0 _._ 005) **0** _._ ( **0** _._ )

PatchTST ND 0 _._ 247(0 _._ 059) 0 _._ 090(0 _._ 002) 0 _._ 080(0 _._ 003) 0 _._ 159(0 _._ 004) **0** _._ ( **0** _._ ) 0 _._ 063(0 _._ 031)
+Ours NRMSE 0 _._ 450(0 _._ 095) **0** _._ ( **0** _._ ) 0 _._ 656(0 _._ 058) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 615(0 _._ 061)


TimeMixer ND **0** _._ ( **0** _._ ) 0 _._ 120(0 _._ 004) 0 _._ 382(0 _._ 011) **0** _._ ( **0** _._ ) 0 _._ 030(0 _._ 014) **0** _._ ( **0** _._ )
NRMSE **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 3 _._ 656(0 _._ 002) 0 _._ 764(0 _._ 003) 0 _._ 041(0 _._ 019) 0 _._ 825(0 _._ 083)


TimeMixer ND 0 _._ 461(0 _._ 021) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) 0 _._ 499(0 _._ 001) **0** _._ ( **0** _._ ) 0 _._ 157(0 _._ 007)
+Ours NRMSE 0 _._ 909(0 _._ 084) 0 _._ 590(0 _._ 530) **3** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ ) **0** _._ ( **0** _._ )


B.3 ADDITIONAL EXPERIMENTAL RESULTS


Table 5 reports the performance of DualRes in mean forecasting, evaluated using the metrics ND
and NRMSE. Although the primary goal of DualRes is to improve probabilistic forecasting, the
framework also enhances mean forecasting performance and increases the stability of predictive
algorithms. We attribute this improvement to the iterative updates in equation 4 of Algorithm 2: since
_F_ - is a nonlinear function, adding the residuals _**ζ**_ _T_ _[∗]_ + _j_ [and applying repeated function compositions]
alter the distributions—and consequently the means—of the pseudo-samples at future steps.


20